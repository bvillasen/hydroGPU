
extern "C"   // ensure functions name to be exactly the same as below
{
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	__device__ void writeBound(  const int boundAxis, const int nCells, double *cnsv, double *bound,
                  const int t_j, const int t_i, const int t_k, const int tid ){
    int boundId, stride;
    if ( boundAxis == 1 ){    //X BOUNDERIES
			boundId = t_i + t_k*N_H;
			stride = N_H*N_D;
		}
    if ( boundAxis == 2 ) {   //Y BOUNDERIES
			boundId = t_j + t_k*N_W;
			stride = N_W*N_D;
		}
    if ( boundAxis == 3 ) {  //Z BOUNDERIES
			boundId = t_j + t_i*N_W;
			stride = N_W*N_H;
		}
    bound[0*stride + boundId] = cnsv[0*nCells + tid];
    bound[1*stride + boundId] = cnsv[1*nCells + tid];
    bound[2*stride + boundId] = cnsv[2*nCells + tid];
    bound[3*stride + boundId] = cnsv[3*nCells + tid];
    bound[4*stride + boundId] = cnsv[4*nCells + tid];
  }

  __global__ void setBounderies( const int nCells, double *cnsv,
				 double* bound_l, double* bound_r, double* bound_d, double* bound_u, double* bound_b, double *bound_t ){
    int t_j = blockIdx.x*blockDim.x + threadIdx.x;
    int t_i = blockIdx.y*blockDim.y + threadIdx.y;
    int t_k = blockIdx.z*blockDim.z + threadIdx.z;
    int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

    bool boundBlock = false;
    if ( blockIdx.x==0 || blockIdx.y==0 || blockIdx.z==0 ) boundBlock = true;
    if ( blockIdx.x==(gridDim.x-1) || blockIdx.y==(gridDim.y-1) || blockIdx.z==(gridDim.z-1) ) boundBlock = true;

    if ( !boundBlock ) return;

    if ( t_j==0 )       writeBound( 1, nCells, cnsv, bound_l, t_j, t_i, t_k, tid );
    if ( t_j==(N_W-1) ) writeBound( 1, nCells, cnsv, bound_r, t_j, t_i, t_k, tid );

    if ( t_i==0 )       writeBound( 2, nCells, cnsv, bound_d, t_j, t_i, t_k, tid );
    if ( t_i==(N_H-1) ) writeBound( 2, nCells, cnsv, bound_u, t_j, t_i, t_k, tid );

    if ( t_k==0 )       writeBound( 3, nCells, cnsv, bound_b, t_j, t_i, t_k, tid );
    if ( t_k==(N_D-1) ) writeBound( 3, nCells, cnsv, bound_t, t_j, t_i, t_k, tid );
  }
}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
