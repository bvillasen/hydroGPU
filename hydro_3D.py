import sys, time, os
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
#import pycuda.curandom as curandom
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
import h5py as h5
import matplotlib.pyplot as plt

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
volumeRenderDirectory = parentDirectory + "/volumeRender"
sys.path.extend( [toolsDirectory, volumeRenderDirectory] )
from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray, kernelMemoryInfo
from tools import ensureDirectory, printProgressTime

cudaP = "double"
nPoints = 128
useDevice = None
usingAnimation = False
showKernelMemInfo = False
usingGravity = False

for option in sys.argv:
  if option == "grav": usingGravity = True
  if option == "float": cudaP = "float"
  if option == "anim": usingAnimation = True
  if option == "mem": showKernelMemInfo = True
  if option == "128" or option == "256": nPoints = int(option)
  if option.find("dev=") != -1: useDevice = int(option[-1])
precision  = {"float":(np.float32, np.complex64), "double":(np.float64,np.complex128) }
cudaPre, cudaPreComplex = precision[cudaP]

print '\nHydro-GPU\n'

#Set output directory
outDir = "/home/bruno/Desktop/data/hydro/"
ensureDirectory( outDir )

#set simulation volume dimentions
nWidth  = nPoints
nHeight = nPoints
nDepth  = nPoints
nFields = 5
nGhost  = 1
nCells  = nWidth*nHeight*nDepth
nCells_b1 = (nWidth+1)*(nHeight+1)*(nDepth+1)
nCells_b2 = (nWidth+2)*(nHeight+2)*(nDepth+2)
# nCells_Ghost = (nWidth+nGhost)*(nHeight+nGhost)*(nDepth+nGhost)
Lx = 1.
Ly = 1.
Lz = 1.
xMax, xMin = Lx/2, -Lx/2
yMax, yMin = Ly/2, -Ly/2
zMax, zMin = Lz/2, -Lz/2
dx, dy, dz = Lx/(nWidth-1), Ly/(nHeight-1), Lz/(nDepth-1 )
Z, Y, X = np.mgrid[ zMin:zMax:nDepth*1j, yMin:yMax:nHeight*1j, xMin:xMax:nWidth*1j ]
xPoints = X[0,0,:]
yPoints = Y[0,:,0]
zPoints = Z[:,0,0]
R = np.sqrt( X*X + Y*Y + Z*Z )
sphereR = 0.25
sphereOffCenter = 0.
sphere = np.sqrt( (X- sphereOffCenter)**2 + Y*Y + Z*Z ) < 0.2
sphere_left  = ( np.sqrt( (X+sphereOffCenter)*(X+sphereOffCenter) + Y*Y + Z*Z ) < sphereR )
sphere_right = ( np.sqrt( (X-sphereOffCenter)*(X-sphereOffCenter) + Y*Y + Z*Z ) < sphereR )
spheres = sphere_right + sphere_left

gamma = 7./5.
c0 = 0.49

#Change precision of the parameters
gamma = cudaPre(gamma)
dx, dy, dz = cudaPre(dx), cudaPre(dy), cudaPre(dz)
Lx, Ly, Lz = cudaPre(Lx), cudaPre(Ly), cudaPre(Lz)
xMin, yMin, zMin = cudaPre(xMin), cudaPre(yMin), cudaPre(zMin)
pi4 = cudaPre( 4*np.pi )

#Initialize openGL
if usingAnimation:
  import volumeRender
  volumeRender.nWidth = nWidth
  volumeRender.nHeight = nHeight
  volumeRender.nDepth = nDepth
  volumeRender.windowTitle = "Hydro 3D  nPoints={0}".format(nPoints)
  volumeRender.initGL()

#initialize pyCUDA context
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=usingAnimation)

#set thread grid for CUDA kernels
block_size_x, block_size_y, block_size_z = 32,2,2   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
gridz = nDepth // block_size_z + 1 * ( nDepth % block_size_z != 0 )
block3D = (block_size_x, block_size_y, block_size_z)
grid3D = (gridx, gridy, gridz)
nBlocks3D = grid3D[0]*grid3D[1]*grid3D[2]
grid3D_poisson = (gridx//2, gridy, gridz)
nPointsBlock = block3D[0]*block3D[1]*block3D[2]
nBlocksGrid = gridx * gridy * gridz
# block2D = ( 16, 16, 1 )
# grid2D = ( nWidth/block2D[0],  nHeight/block2D[1], 1 )
block1D = (512,1,1)
grid1D = (nCells/block1D[0], 1, 1)

print "\nCompiling CUDA code"
cudaCodeFile = open("cuda_hydro_3D.cu","r")
cudaCodeString = cudaCodeFile.read().replace( "cudaP", cudaP )
cudaCodeString = cudaCodeString.replace('N_WIDTH', str(nWidth) )
cudaCodeString = cudaCodeString.replace('N_HEIGHT', str(nHeight) )
cudaCodeString = cudaCodeString.replace('N_DEPTH', str(nDepth) )
cudaCodeString = cudaCodeString.replace( "THREADS_PER_BLOCK", str(nPointsBlock) )
cudaCode = SourceModule(cudaCodeString, include_dirs=[currentDirectory])
setBounderies_knl = cudaCode.get_function('setBounderies')
setInterFlux_hll_knl = cudaCode.get_function('setInterFlux_hll')
getInterFlux_hll_knl = cudaCode.get_function('getInterFlux_hll')
addDtoD_knl = cudaCode.get_function('addDtoD')
convertToUCHAR = cudaCode.get_function('convertToUCHAR')
reduction_min_kernel = cudaCode.get_function('reduction_min_kernel')
reduction_max_kernel = cudaCode.get_function('reduction_max_kernel')
reconstruct_PCM_knl = cudaCode.get_function('reconstruct_PCM' )
solveRiemann_knl = cudaCode.get_function('solveRiemann' )
setTimes_knl = cudaCode.get_function('setTimes')
advanceConserved_knl = cudaCode.get_function('advanceConserved')

def timeStepHydro():
  setBounderies_knl( np.int32(nCells), cnsv_d,
      bound_l_d, bound_r_d, bound_d_d, bound_u_d, bound_b_d, bound_t_d, grid=grid3D, block=block3D  )

  setTimes_knl( np.int32(nCells), gamma, dx, dy, dz, cnsv_d, times_d, grid=grid3D, block=block3D )
  dt = c0 * reduction_min( times_d, prePartialSum_d, partialSum_h, partialSum_d )

  reconstruct_PCM_knl( np.int32(nCells), np.int32(nCells_b2), cnsv_d, recons_l, recons_r, grid=grid3D, block=block3D)

  for coord in [ 1, 2, 3]:
    if coord == 1:
      iFlx_temp = iFlx_x_d
      # bound_l_temp, bound_r_temp = bound_l_d, bound_r_d
      # iFlx_bound_temp = iFlx_bnd_r_d
    if coord == 2:
      iFlx_temp = iFlx_y_d
      # bound_l_temp, bound_r_temp = bound_d_d, bound_u_d
      # iFlx_bound_temp = iFlx_bnd_u_d
    if coord == 3:
      iFlx_temp = iFlx_z_d
      # bound_l_temp, bound_r_temp = bound_b_d, bound_t_d
      # iFlx_bound_temp = iFlx_bnd_t_d

    solveRiemann_knl( np.int32( coord ), np.int32(nCells), np.int32(nCells_b1), np.int32(nCells_b2),
      gamma, dx, dy, dz, recons_l, recons_r, iFlx_temp, grid=grid3D, block=block3D )

  #   setInterFlux_hll_knl( np.int32( coord ), np.int32(nCells), gamma, dx, dy, dz,
  #     cnsv_d, iFlx_d,
  #     bound_l_temp, bound_r_temp,
  #     iFlx_bound_temp, times_d,  grid=grid3D, block=block3D )
  #
  #   getInterFlux_hll_knl( np.int32( coord ),  np.int32(nCells), cudaPre(dt), gamma, dx, dy, dz,
  #       cnsv_adv_d, iFlx_d, iFlx_bound_temp, grid=grid3D, block=block3D )
  # addDtoD_knl( np.int32(nCells), cnsv_d, cnsv_adv_d, grid=grid3D, block=block3D)

  advanceConserved_knl( np.int32(nCells), np.int32(nCells_b1),
      cudaPre(dt), gamma, dx, dy, dz, cnsv_d, iFlx_x_d, iFlx_y_d, iFlx_z_d, grid=grid3D, block=block3D )

def stepFuntion():
  # maxVal = ( gpuarray.max( cnsv1_d ) ).get()
  maxVal = reduction_max( cnsv_d, prePartialSum_d, partialSum_h, partialSum_d )
  # maxVal = 1.
  # convertToUCHAR_old( np.int32(0), np.int32(nCells), cudaPre( 0.95/maxVal ), cnsv1_d, plotData_d)
  convertToUCHAR( np.int32(0), np.int32(nCells), cudaPre( 0.95/maxVal ), cnsv_d, plotData_d, grid=grid1D, block=block1D)
  copyToScreenArray()
  timeStepHydro()
  # if usingGravity: getGravForce()

########################################################################
if showKernelMemInfo:
  #kernelMemoryInfo( setFlux_kernel, 'setFlux_kernel')
  #print ""
  kernelMemoryInfo( setInterFlux_hll_kernel, 'setInterFlux_hll_kernel')
  print ""
  kernelMemoryInfo( getInterFlux_hll_kernel, 'getInterFlux_hll_kernel')
  print ""
  kernelMemoryInfo( iterPoissonStep_kernel, 'iterPoissonStep_kernel')
  print ""
  kernelMemoryInfo( getBounderyPotential_kernel, 'getBounderyPotential_kernel')
  print ""
  kernelMemoryInfo( reduceDensity_kernel, 'reduceDensity_kernel')
  print ""
########################################################################
########################################################################
print "\nInitializing Data"
initialMemory = getFreeMemory( show=True )
rho = np.zeros( X.shape, dtype=cudaPre )  #density
vx  = np.zeros( X.shape, dtype=cudaPre )
vy  = np.zeros( X.shape, dtype=cudaPre )
vz  = np.zeros( X.shape, dtype=cudaPre )
p   = np.zeros( X.shape, dtype=cudaPre )  #pressure
#####################################################
#Initialize a centerd sphere
overDensity = sphere
rho[ overDensity ] = 1.
rho[ np.logical_not(overDensity) ] = 0.6
overPresure = sphere
p[ overPresure ] = 10
p[ np.logical_not(overPresure) ] = 1
v2 = vx*vx + vy*vy + vz*vz
#####################################################
#Initialize conserved values
cnsv1_h = rho
cnsv2_h = rho * vx
cnsv3_h = rho * vy
cnsv4_h = rho * vz
cnsv5_h = rho*v2/2. + p/(gamma-1)
cnsv_h = np.array([ cnsv1_h, cnsv2_h, cnsv3_h, cnsv4_h, cnsv5_h ])
#Arrays for bounderies
ZEROS_HD, ZEROS_WD, ZEROS_WH = np.zeros( [ nFields, nHeight, nDepth ], dtype=cudaPre ), np.zeros( [ nFields, nWidth, nDepth ], dtype=cudaPre ), np.zeros( [ nFields, nWidth, nHeight ], dtype=cudaPre )
bound_l_h = np.zeros_like( ZEROS_HD )
bound_r_h = np.zeros_like( ZEROS_HD )
bound_d_h = np.zeros_like( ZEROS_WD )
bound_u_h = np.zeros_like( ZEROS_WD )
bound_b_h = np.zeros_like( ZEROS_WH )
bound_t_h = np.zeros_like( ZEROS_WH )
#####################################################
#Initialize device global data
cnsv_d = gpuarray.to_gpu( cnsv_h )
# cnsv_adv_d = gpuarray.to_gpu( cnsv_h )
# ZEROS = np.zeros( X.shape, dtype=cudaPre )
ZEROS_all   = np.zeros( [nFields, nDepth, nHeight, nWidth], dtype=cudaPre )
ZEROS_GHOST = np.zeros( [nFields, nWidth+2*nGhost, nHeight+2*nGhost, nDepth+2*nGhost ], dtype=cudaPre )
ZEROS_b2    = np.zeros( [nFields, nWidth+2, nHeight+2, nDepth+2 ], dtype=cudaPre )
times_d = gpuarray.to_gpu( np.zeros( X.shape, dtype=cudaPre ) )
# iFlx_d = gpuarray.to_gpu( ZEROS_all )
iFlx_x_d = gpuarray.to_gpu( np.zeros( [nFields, nWidth+1, nHeight+1, nDepth+1 ], dtype=cudaPre ) )
iFlx_y_d = gpuarray.to_gpu( np.zeros( [nFields, nWidth+1, nHeight+1, nDepth+1 ], dtype=cudaPre ) )
iFlx_z_d = gpuarray.to_gpu( np.zeros( [nFields, nWidth+1, nHeight+1, nDepth+1 ], dtype=cudaPre ) )
bound_l_d, bound_r_d = gpuarray.to_gpu( bound_l_h ), gpuarray.to_gpu( bound_r_h )
bound_d_d, bound_u_d = gpuarray.to_gpu( bound_d_h ), gpuarray.to_gpu( bound_u_h )
bound_b_d, bound_t_d = gpuarray.to_gpu( bound_b_h ), gpuarray.to_gpu( bound_t_h )
recons_l, recons_r = gpuarray.to_gpu( ZEROS_b2 ), gpuarray.to_gpu( ZEROS_b2 )
# iFlx_bnd_r_d, iFlx_bnd_u_d, iFlx_bnd_t_d = gpuarray.to_gpu( ZEROS_HD ), gpuarray.to_gpu( ZEROS_WD ), gpuarray.to_gpu( ZEROS_WH )
#Arrays for reductions
blockSize_reduc = 512
gridSize_reduc = nCells / blockSize_reduc  / 2
last_gridSize = gridSize_reduc / blockSize_reduc / 2
prePartialSum_d = gpuarray.to_gpu( np.zeros( gridSize_reduc, dtype=cudaPre ) )
partialSum_h = np.zeros( last_gridSize, dtype=cudaPre )
partialSum_d = gpuarray.to_gpu( partialSum_h )

def reduction_min( data_d, prePartialSum_d, partialSum_h, partialSum_d ):
  reduction_min_kernel(  data_d, prePartialSum_d , grid=(gridSize_reduc,1,1) , block=(blockSize_reduc,1,1) )
  reduction_min_kernel( prePartialSum_d, partialSum_d, grid=(last_gridSize,1,1), block=(blockSize_reduc,1,1) )
  partialSum_h = partialSum_d.get()
  return  partialSum_h.min()

def reduction_max( data_d, prePartialSum_d, partialSum_h, partialSum_d ):
  reduction_max_kernel(  data_d, prePartialSum_d , grid=(gridSize_reduc,1,1) , block=(blockSize_reduc,1,1) )
  reduction_max_kernel( prePartialSum_d, partialSum_d, grid=(last_gridSize,1,1), block=(blockSize_reduc,1,1) )
  partialSum_h = partialSum_d.get()
  return  partialSum_h.max()

if usingAnimation:
  plotData_d = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
  volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotData_d )
print "Total Global Memory Used: {0:.2f} MB\n".format(float(initialMemory-getFreeMemory( show=False ))/1e6)

#configure volumeRender functions
if usingAnimation:
  #volumeRender.viewTranslation[2] = -2
  volumeRender.transferScale = np.float32( 2.8 )
  #volumeRender.keyboard = keyboard
  #volumeRender.specialKeys = specialKeyboardFunc
  volumeRender.stepFunc = stepFuntion
  #run volumeRender animation
  volumeRender.animate()


#Set output file
outFileName = "hydro_data.h5"
outFile = h5.File( outDir + outFileName, "w")
print 'outDir: {0}'.format( outDir )
outFile.close()
