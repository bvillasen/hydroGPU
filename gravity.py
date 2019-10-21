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
nPoints = 256
useDevice = None
usingAnimation = False

for option in sys.argv:
  if option == "float": cudaP = "float"
  if option == "anim": usingAnimation = True
  if option == "mem": showKernelMemInfo = True
  if option == "128" or option == "256": nPoints = int(option)
  if option.find("dev=") != -1: useDevice = int(option[-1])
precision  = {"float":(np.float32, np.complex64), "double":(np.float64,np.complex128) }
cudaPre, cudaPreComplex = precision[cudaP]

#set simulation volume dimentions
nWidth = nPoints
nHeight = nPoints
nDepth = nPoints
nData = nWidth*nHeight*nDepth

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
zPoints = Z[0,0,:]
R = np.sqrt( X*X + Y*Y + Z*Z )
sphereR = 0.1
sphereOffCenter = 0.25
sphere = np.sqrt( X*X + Y*Y + Z*Z ) < sphereR
sphere_left  = ( np.sqrt( (X+sphereOffCenter)*(X+sphereOffCenter) + Y*Y + Z*Z ) < sphereR )
sphere_right = ( np.sqrt( (X-sphereOffCenter)*(X-sphereOffCenter) + Y*Y + Z*Z ) < sphereR )
spheres = sphere_right + sphere_left


#For analitical solution
sigma = 0.2
r2 = X*X + Y*Y + Z*Z
rho_teo = ( r2 - 2*sigma**2 )/sigma**4 * np.exp( -r2/(2*sigma**2) )
phi_teo = np.exp( -r2/(2*sigma**2) )



#Change precision of the parameters
dx, dy, dz = cudaPre(dx), cudaPre(dy), cudaPre(dz)
# Lx, Ly, Lz = cudaPre(Lx), cudaPre(Ly), cudaPre(Lz)
# xMin, yMin, zMin = cudaPre(xMin), cudaPre(yMin), cudaPre(zMin)
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
block_size_x, block_size_y, block_size_z = 32,4,4   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
gridz = nDepth // block_size_z + 1 * ( nDepth % block_size_z != 0 )
block3D = (block_size_x, block_size_y, block_size_z)
grid3D = (gridx, gridy, gridz)
nBlocks3D = grid3D[0]*grid3D[1]*grid3D[2]
grid3D_poisson = (gridx//2, gridy, gridz)


print "\nCompiling CUDA code"
cudaCodeFile = open("cuda_gravity.cu","r")
cudaCodeString = cudaCodeFile.read().replace( "cudaP", cudaP )
# cudaCodeString = cudaCodeString.replace( "THREADS_PER_BLOCK", str(nPointsBlock) )
cudaCode = SourceModule(cudaCodeString)
iterPoissonStep_kernel = cudaCode.get_function('iterPoissonStep')
FFT_divideK2_kernel = cudaCode.get_function('FFT_divideK2_kernel')
########################################################################
convertToUCHAR = ElementwiseKernel(arguments="cudaP normaliztion, cudaP *values, unsigned char *psiUCHAR".replace("cudaP", cudaP),
			      operation = "psiUCHAR[i] = (unsigned char) ( -255*( values[i]*normaliztion -1 ) );",
			      name = "sendModuloToUCHAR_kernel")

def poisonIteration( parity, omega ):
  iterPoissonStep_kernel( np.int32(parity),
  np.int32( nWidth ), np.int32( nHeight ), np.int32( nDepth ),
  dx, dy, dz, cudaPre(omega), pi4,
  rho_d, phi_d, converged, grid=grid3D_poisson, block=block3D  )

rJacobi = ( np.cos(np.pi/nWidth) + (dx/dy)**2*np.cos(np.pi/nHeight) ) / ( 1 + (dx/dy)**2 )
def poissonStep( omega ):
  converged.set( one_Array )
  poisonIteration( 0, omega )
  poisonIteration( 1, omega )
  hasConverged = converged.get()[0]
  return hasConverged

########################################################################
def solvePoisson( show=False ):
  maxIter = 500000
  omega = 2. / ( 1 + np.pi / nWidth  )
  # omega = 1
  for n in range(maxIter):
    hasConverged = poissonStep( omega )
    if hasConverged == 1:
      phi_1 = phi_d.get()
      poisonIteration( 0, omega )
      phi_2 = phi_d.get()
      phi_avrg = ( phi_1 + phi_2 )/2.
      if show: print 'Poisson converged: ', n+1
      # return phi_1, phi_2, phi_avrg
      return phi_avrg
  if show: print 'Poisson converged: ', maxIter
  return phi_d.get()

########################################################################
# def stepFuntion():
#   solvePoisson()
#   maxVal = ( gpuarray.max( phi_d ) ).get()
#   convertToUCHAR( cudaPre( 0.95/maxVal ), phi_d, plotData_d)
#   copyToScreenArray()
#
#   # grid=grid3D_poisson, block=block3D

########################################################################
########################################################################
print "\nInitializing Data"
initialMemory = getFreeMemory( show=True )
rho = np.zeros( X.shape, dtype=cudaPre )  #density
#####################################################
#Initialize a centerd sphere
overDensity = spheres
rho[ overDensity ] = 1.
rho[ np.logical_not(overDensity) ] = 0.6
rho = rho_teo
# phi = np.ones( X.shape, dtype=cudaPre )   #gravity potencial
phi = rho   #gravity potencial
zeros_h = np.zeros_like( rho )
#####################################################
#Initialize device global data
phi_d = gpuarray.to_gpu( phi )
rho_d = gpuarray.to_gpu( rho )
rho_re_d = gpuarray.to_gpu( rho )
rho_im_d = gpuarray.to_gpu( zeros_h )
rho_FFT_re_d = gpuarray.to_gpu( zeros_h )
rho_FFT_im_d = gpuarray.to_gpu(zeros_h)
# phiTemp_d = gpuarray.to_gpu( phi )
one_Array = np.array([ 1 ]).astype( np.int32 )
converged = gpuarray.to_gpu( one_Array )
if usingAnimation:
  plotData_d = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
  volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotData_d )
print "Total Global Memory Used: {0:.2f} MB\n".format(float(initialMemory-getFreeMemory( show=False ))/1e6)



print " Making FFT plan"
from pyfft.cuda import Plan
fftPlan = Plan((nDepth, nHeight, nWidth),  dtype=cudaPre)
fftKx_h = np.zeros( nWidth, dtype=cudaPre )
fftKy_h = np.zeros( nHeight, dtype=cudaPre )
fftKz_h = np.zeros( nDepth, dtype=cudaPre )
for i in range(1,nWidth/2+1):
  fftKx_h[i-1] = i*2*np.pi/Lx
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
fftKy_d = gpuarray.to_gpu( fftKx_h )
fftKz_d = gpuarray.to_gpu( fftKx_h )


fftPlan.execute( rho_re_d, rho_im_d, rho_FFT_re_d, rho_FFT_im_d )
FFT_divideK2_kernel( fftKx_d, fftKy_d, fftKz_d, rho_FFT_re_d, rho_FFT_im_d, grid=grid3D, block=block3D )
fftPlan.execute( rho_FFT_re_d, rho_FFT_im_d, rho_re_d, rho_im_d,  inverse=True )
phi_re_FFT = rho_re_d.get()
phi_im_FFT = rho_im_d.get()
# phi_FFT = np.sqrt( phi_re_FFt**2 + phi_im_FFt**2 )


phi_slide_teo = phi_teo[nWidth/2,:,:]
plt.figure(0)
plt.clf()
plt.imshow(phi_slide_teo, interpolation='nearest')
plt.colorbar()
plt.show()

#
# phi_slide_FFT = phi_re_FFT[nWidth/2,:,:]
# plt.figure(1)
# plt.imshow(phi_slide_FFT, interpolation='nearest')
# plt.colorbar()
# plt.show()



print 'Getting initial Gravity Force...'
start, end = cuda.Event(), cuda.Event()
start.record() # start timing
# phi_1, phi_2, phi = solvePoisson( show=True )
phi = solvePoisson( show=True )
phi = phi - phi.min()
phi = phi/phi.max()
end.record(), end.synchronize()
secs = start.time_till( end )*1e-3
print 'Time: {0:0.4f}\n'.format( secs )


# phi_slide_1, phi_slide_2, phi_slide = phi_1[nWidth/2,:,:], phi_2[nWidth/2,:,:], phi[nWidth/2,:,:]
phi_slide = phi[nWidth/2,:,:]
plt.figure(2)
plt.clf()
plt.imshow(phi_slide, interpolation='nearest')
plt.colorbar()
plt.show()

#
#
plt.figure(3)
plt.clf()
plt.imshow(np.abs(phi_slide - phi_slide_teo), interpolation='nearest')
plt.colorbar()
plt.show()



# # plt.figure(1)
# # plt.imshow(phi_slide_1, interpolation='nearest')
# # plt.figure(2)
# # plt.imshow(phi_slide_2, interpolation='nearest')






#configure volumeRender functions
if usingAnimation:
  #volumeRender.viewTranslation[2] = -2
  # volumeRender.transferScale = np.float32( 2.8 )
  #volumeRender.keyboard = keyboard
  #volumeRender.specialKeys = specialKeyboardFunc
  volumeRender.stepFunc = stepFuntion
  #run volumeRender animation
  volumeRender.animate()
