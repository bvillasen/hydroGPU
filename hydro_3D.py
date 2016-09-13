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
from pyfft.cuda import Plan

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
sphere = np.sqrt( (X- sphereOffCenter)**2 + Y*Y + Z*Z ) < sphereR
sphere_left  = ( np.sqrt( (X+sphereOffCenter)*(X+sphereOffCenter) + Y*Y + Z*Z ) < sphereR )
sphere_right = ( np.sqrt( (X-sphereOffCenter)*(X-sphereOffCenter) + Y*Y + Z*Z ) < sphereR )
spheres = sphere_right + sphere_left

gamma = 7./5.
c0 = 0.4

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
TperB = 512
block1D = (TperB,1,1)
grid1D = (nCells/block1D[0], 1, 1)

print "\nCompiling CUDA code"
hydroCodeFile = open("cuda_hydro_3D.cu","r")
cudaCodeString = hydroCodeFile.read().replace( "cudaP", cudaP )
cudaCodeString = cudaCodeString.replace('N_WIDTH', str(nWidth) )
cudaCodeString = cudaCodeString.replace('N_HEIGHT', str(nHeight) )
cudaCodeString = cudaCodeString.replace('N_DEPTH', str(nDepth) )
cudaCodeString = cudaCodeString.replace( "THREADS_PER_BLOCK", str(TperB) )
cudaCode = SourceModule(cudaCodeString, include_dirs=[currentDirectory])
setBounderies_knl = cudaCode.get_function('setBounderies')
setInterFlux_hll_knl = cudaCode.get_function('setInterFlux_hll')
getInterFlux_hll_knl = cudaCode.get_function('getInterFlux_hll')
addDtoD_knl = cudaCode.get_function('addDtoD')
convertToUCHAR = cudaCode.get_function('convertToUCHAR')
reduction_min_kernel = cudaCode.get_function('reduction_min_kernel')
reduction_max_kernel = cudaCode.get_function('reduction_max_kernel')
reconstruct_PCM_knl = cudaCode.get_function('reconstruct_PCM' )
#reconstruct_PLM_knl = cudaCode.get_function('reconstruct_PLM' )
solveRiemann_knl = cudaCode.get_function('solveRiemann' )
setTimes_knl = cudaCode.get_function('setTimes')
advanceConserved_knl = cudaCode.get_function('advanceConserved')
advanceTransverseFlux_knl = cudaCode.get_function('advanceTransverseFlux')
# gravityCodeFile = open("cuda_gravity_3D.cu","r")
# cudaCodeString = gravityCodeFile.read().replace( "cudaP", cudaP )
# cudaCodeString = cudaCodeString.replace('N_WIDTH', str(nWidth) )
# cudaCodeString = cudaCodeString.replace('N_HEIGHT', str(nHeight) )
# cudaCodeString = cudaCodeString.replace('N_DEPTH', str(nDepth) )
# cudaCodeString = cudaCodeString.replace( "THREADS_PER_BLOCK", str(TperB) )
# cudaCode = SourceModule(cudaCodeString, include_dirs=[currentDirectory])
iterPoissonStep_knl = cudaCode.get_function('iterPoissonStep')
getGravityForce_knl = cudaCode.get_function('getGravityForce')
copyDensity_knl = cudaCode.get_function('copyDensity')
FFT_divideK2_knl = cudaCode.get_function('FFT_divideK2')
def poissonHalfStep( parity, omega, convEpsilon ):
  iterPoissonStep_knl( np.int32(parity), np.int32(nWidth), np.int32(nHeight),np.int32(nDepth),
  dx, dy, dz, omega, pi4, convEpsilon, cnsv_d, phi_d, converged_d, grid=grid3D_poisson, block=block3D )


def iterPoissonStep( omega, convEpsilon ):
  poissonHalfStep( 0, omega, convEpsilon )
  poissonHalfStep( 1, omega, convEpsilon )

def poissonMultiStep( omega, nStepsPerIter, convEpsilon ):
  converged_d.set( one_Array )
  [ iterPoissonStep( omega, convEpsilon ) for i in range(nStepsPerIter)]
  hasConverged = converged_d.get()[0]
  return hasConverged

nStepsPerIter = 1
def solvePoisson( convEpsilon, copy=False, show=False ):
  maxIter = 50000
  omega = np.float64( 2. / ( 1 + np.pi / nWidth  ) )
  # omega = np.float64( 1.99 )
  # omega = 1
  for n in range(maxIter):
    hasConverged = poissonMultiStep( omega, nStepsPerIter, convEpsilon )
    if hasConverged == 1:
      if show: print ' Poisson converged: {0}'.format( (n+1)*nStepsPerIter )
      if copy: return phi_d.get()
      else: return
  if show: print 'Poisson NOT-converged\n'
  return phi_d.get()

def getGravityForce():
  getGravityForce_knl( np.int32(nCells), np.int32(nWidth), np.int32(nHeight), np.int32(nDepth),
  dx, dy, dz, gForceX_d, gForceY_d, gForceZ_d, cnsv_d, phi_d, gravWork_d, block=block3D, grid=grid3D  )

def solveGravity( convEpsilon ):
  global time_grav, start_grav, end_grav
  start_grav.record()
  solvePoisson( convEpsilon, show=True)
  getGravityForce()
  end_grav.record(), end_grav.synchronize()
  time_grav += start_grav.time_till( end_grav )*1e-3

def timeStepHydro( convEpsilon ):
  global time_hydro, start_hydro, end_hydro
  start_hydro.record()
  setBounderies_knl( np.int32(nCells), cnsv_d,
      bound_l_d, bound_r_d, bound_d_d, bound_u_d, bound_b_d, bound_t_d, grid=grid3D, block=block3D  )

  setTimes_knl( np.int32(nCells), gamma, dx, dy, dz, cnsv_d, times_d, grid=grid3D, block=block3D )
  # dt = c0 * reduction_min( times_d, prePartialSum_d, partialSum_h, partialSum_d )
  dt = c0 * gpuarray.min(times_d).get()

  for coord in [ 1, 2, 3]:
    if coord == 1:
      recons_l, recons_r = recons_l_x_d, recons_r_x_d
      bound_l, bound_r = bound_l_d, bound_r_d
      iFlx_temp = iFlx_x_d
    if coord == 2:
      recons_l, recons_r = recons_l_y_d, recons_r_y_d
      bound_l, bound_r = bound_d_d, bound_u_d
      iFlx_temp = iFlx_y_d
    if coord == 3:
      recons_l, recons_r = recons_l_z_d, recons_r_z_d
      bound_l, bound_r = bound_b_d, bound_t_d
      iFlx_temp = iFlx_z_d
    # reconstruct_PLM_knl( np.int32(coord), np.int32(nCells), np.int32(nCells_b2), cnsv_d, bound_l, bound_r, recons_l, recons_r, grid=grid3D, block=block3D )
    reconstruct_PCM_knl( np.int32(nCells), np.int32(nCells_b2), cnsv_d, recons_l, recons_r, grid=grid3D, block=block3D)
    solveRiemann_knl( np.int32( coord ), np.int32(nCells), np.int32(nCells_b1), np.int32(nCells_b2),
      gamma, dx, dy, dz, recons_l, recons_r, iFlx_temp, grid=grid3D, block=block3D )


  for coord in [ 1, 2, 3]:
    if coord == 1:
      recons_l, recons_r = recons_l_x_d, recons_r_x_d
      delta_1, delta_2 = 0.5*dt/dy, 0.5*dt/dz
      iFlx_trans_1, iFlx_trans_2 = iFlx_y_d, iFlx_z_d
    if coord == 2:
      recons_l, recons_r = recons_l_y_d, recons_r_y_d
      delta_1, delta_2 = 0.5*dt/dz, 0.5*dt/dx
      iFlx_trans_1, iFlx_trans_2 = iFlx_z_d, iFlx_x_d
    if coord == 3:
      recons_l, recons_r = recons_l_z_d, recons_r_z_d
      delta_1, delta_2 = 0.5*dt/dx, 0.5*dt/dy
      iFlx_trans_1, iFlx_trans_2 = iFlx_x_d, iFlx_y_d
    advanceTransverseFlux_knl( np.int32(coord), np.int32(nCells), np.int32(nCells_b1),  np.int32(nCells_b2),
        cudaPre(delta_1), cudaPre(delta_2), recons_l, recons_r,
        iFlx_trans_1, iFlx_trans_2, grid=grid3D, block=block3D)

  for coord in [ 1, 2, 3]:
    if coord == 1:
      recons_l, recons_r = recons_l_x_d, recons_r_x_d
      iFlx_temp = iFlx_x_d
    if coord == 2:
      recons_l, recons_r = recons_l_y_d, recons_r_y_d
      iFlx_temp = iFlx_y_d
    if coord == 3:
      recons_l, recons_r = recons_l_z_d, recons_r_z_d
      iFlx_temp = iFlx_z_d
    solveRiemann_knl( np.int32( coord ), np.int32(nCells), np.int32(nCells_b1), np.int32(nCells_b2),
      gamma, dx, dy, dz, recons_l, recons_r, iFlx_temp, grid=grid3D, block=block3D )

  # convEpsilon = np.float64( 0.00005 )
  # if usingGravity: solveGravity( convEpsilon )
  advanceConserved_knl( np.int32(nCells), np.int32(nCells_b1),
      cudaPre(dt), dx, dy, dz, cnsv_d, iFlx_x_d, iFlx_y_d, iFlx_z_d,
      gForceX_d, gForceY_d, gForceZ_d, gravWork_d, grid=grid3D, block=block3D )
  end_hydro.record(), end_hydro.synchronize()
  time_hydro += start_hydro.time_till( end_hydro )*1e-3


def stepFuntion():
  global convEpsilon
  # maxVal = ( gpuarray.max( cnsv1_d ) ).get()
  maxVal = reduction_max( cnsv_d, prePartialSum_d, partialSum_h, partialSum_d )
  #maxVal = 1.
  # convertToUCHAR_old( np.int32(0), np.int32(nCells), cudaPre( 0.95/maxVal ), cnsv1_d, plotData_d)
  convertToUCHAR( np.int32(0), np.int32(nCells), cudaPre( 0.95/maxVal ), cnsv_d, plotData_d, grid=grid1D, block=block1D)
  copyToScreenArray()
  if usingGravity: solveGravity( convEpsilon )
  timeStepHydro( convEpsilon )
  print 'Time: {0:.2f}  {1:.2f}  {2:.2f}  {3:.2f}'.format( time_hydro+time_grav, time_hydro, time_grav, time_grav/time_hydro )
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
overPresureVal = 10
if usingGravity: overPresureVal = 1
p[ overPresure ] = overPresureVal
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
#Arrays for gravity
phi_h = np.zeros_like( rho )
#####################################################
#Initialize device global data
rho_d  = gpuarray.to_gpu( rho )
cnsv_d = gpuarray.to_gpu( cnsv_h )
# cnsv_adv_d = gpuarray.to_gpu( cnsv_h )
# ZEROS = np.zeros( X.shape, dtype=cudaPre )
ZEROS_all   = np.zeros( [nFields, nDepth, nHeight, nWidth], dtype=cudaPre )
ZEROS_GHOST = np.zeros( [nFields, nWidth+2*nGhost, nHeight+2*nGhost, nDepth+2*nGhost ], dtype=cudaPre )
ZEROS_b2    = np.zeros( [nFields, nWidth+2, nHeight+2, nDepth+2 ], dtype=cudaPre )
times_d = gpuarray.to_gpu( np.zeros( X.shape, dtype=cudaPre ) )
# iFlx_d = gpuarray.to_gpu( ZEROS_all )
recons_l_x_d, recons_r_x_d = gpuarray.to_gpu( ZEROS_b2 ), gpuarray.to_gpu( ZEROS_b2 )
recons_l_y_d, recons_r_y_d = gpuarray.to_gpu( ZEROS_b2 ), gpuarray.to_gpu( ZEROS_b2 )
recons_l_z_d, recons_r_z_d = gpuarray.to_gpu( ZEROS_b2 ), gpuarray.to_gpu( ZEROS_b2 )

# recons_adv_l, recons_adv_r = gpuarray.to_gpu( ZEROS_b2 ), gpuarray.to_gpu( ZEROS_b2 )
iFlx_x_d = gpuarray.to_gpu( np.zeros( [nFields, nWidth+1, nHeight+1, nDepth+1 ], dtype=cudaPre ) )
iFlx_y_d = gpuarray.to_gpu( np.zeros( [nFields, nWidth+1, nHeight+1, nDepth+1 ], dtype=cudaPre ) )
iFlx_z_d = gpuarray.to_gpu( np.zeros( [nFields, nWidth+1, nHeight+1, nDepth+1 ], dtype=cudaPre ) )
bound_l_d, bound_r_d = gpuarray.to_gpu( bound_l_h ), gpuarray.to_gpu( bound_r_h )
bound_d_d, bound_u_d = gpuarray.to_gpu( bound_d_h ), gpuarray.to_gpu( bound_u_h )
bound_b_d, bound_t_d = gpuarray.to_gpu( bound_b_h ), gpuarray.to_gpu( bound_t_h )
# iFlx_bnd_r_d, iFlx_bnd_u_d, iFlx_bnd_t_d = gpuarray.to_gpu( ZEROS_HD ), gpuarray.to_gpu( ZEROS_WD ), gpuarray.to_gpu( ZEROS_WH )
phi_d = gpuarray.to_gpu( phi_h )
one_Array = np.array([ 1 ]).astype( np.int32 )
converged_d = gpuarray.to_gpu( one_Array )
gForceX_d = gpuarray.to_gpu( phi_h )
gForceY_d = gpuarray.to_gpu( phi_h )
gForceZ_d = gpuarray.to_gpu( phi_h )
gravWork_d = gpuarray.to_gpu( phi_h )
#Arrays for FFT calculation
rho_imag_d = gpuarray.to_gpu( np.zeros_like( rho ) )
fftKx_h = np.zeros( nWidth, dtype=cudaPre )
fftKy_h = np.zeros( nHeight, dtype=cudaPre )
fftKz_h = np.zeros( nDepth, dtype=cudaPre )
for i in range(nWidth/2):
  fftKx_h[i] = i*2*np.pi/Lx
for i in range(nWidth/2, nWidth):
  fftKx_h[i] = (i-nWidth)*2*np.pi/Lx
for i in range(nHeight/2):
  fftKy_h[i] = i*2*np.pi/Ly
for i in range(nHeight/2, nHeight):
  fftKy_h[i] = (i-nHeight)*2*np.pi/Ly
for i in range(nDepth/2):
  fftKz_h[i] = i*2*np.pi/Lz
for i in range(nDepth/2, nDepth):
  fftKz_h[i] = (i-nDepth)*2*np.pi/Lz
fftKx_d = gpuarray.to_gpu( fftKx_h )         #OPTIMIZATION
fftKy_d = gpuarray.to_gpu( fftKy_h )
fftKz_d = gpuarray.to_gpu( fftKz_h )


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

#Cuda events
start, end = cuda.Event(), cuda.Event()
start_hydro, end_hydro = cuda.Event(), cuda.Event()
start_grav, end_grav = cuda.Event(), cuda.Event()
time_hydro = 0
time_grav  = 0

# if usingGravity:
#   print 'Making FFT 3D plan'
#   fft_plan = Plan( (nDepth, nHeight, nWidth), dtype=np.float64 )
#   print 'Getting initial gravity force by FFT'
#   start_grav.record()
#   copyDensity_knl( cnsv_d, rho_d, grid=grid3D, block=block3D )
#   fft_plan.execute( rho_d, rho_imag_d )
#   FFT_divideK2_knl( fftKx_d, fftKy_d, fftKz_d, rho_d, rho_imag_d, grid=grid3D, block=block3D )
#   fft_plan.execute( rho_d, rho_imag_d, inverse=True )
#   phi_fft_re = rho_d.get()
#   phi_fft_im = rho_imag_d.get()
#   end_grav.record(), end.synchronize()
#   secs = start_grav.time_till( end_grav )*1e-3
#   print ' Time: {0:.03f} secs\n'.format(  secs )

convEpsilon = np.float64( 0.0001 )
if usingGravity:
  print 'Getting initial gravity force'
  # convEpsilon = np.float64( 0.00005 )
  start.record()
  phi_h = solvePoisson( convEpsilon, copy=True, show=True )
  getGravityForce()
  gForceX, gForceY, gForceZ = gForceX_d.get(), gForceY_d.get(), gForceZ_d.get()
  end.record(), end.synchronize()
  secs = start.time_till( end )*1e-3
  print ' Time: {0:.03f} secs\n'.format(  secs )

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

stride = 1
outFile.create_dataset('rho', data=rho[::stride,::stride,::stride].astype(np.float32))
outFile.create_dataset('phi', data=phi_h[::stride,::stride,::stride].astype(np.float32))
# outFile.create_dataset('phi_fft_re', data=phi_fft_re[::stride,::stride,::stride].astype(np.float32))
# outFile.create_dataset('phi_fft_im', data=phi_fft_im[::stride,::stride,::stride].astype(np.float32))
outFile.create_dataset('gForceX', data=gForceX[::stride,::stride,::stride].astype(np.float32))
outFile.create_dataset('gForceY', data=gForceY[::stride,::stride,::stride].astype(np.float32))
outFile.create_dataset('gForceZ', data=gForceZ[::stride,::stride,::stride].astype(np.float32))


outFile.close()
