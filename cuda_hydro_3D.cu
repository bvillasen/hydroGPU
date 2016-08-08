#include <pycuda-helpers.hpp>

#define N_W N_WIDTH
#define N_H N_HEIGHT
#define N_D N_DEPTH

extern "C"   // ensure functions name to be exactly the same as below
{
	__global__ void convertToUCHAR( const int field, const int nCells, cudaP normaliztion, cudaP *values, unsigned char *psiUCHAR ){
		int tid = blockIdx.x*blockDim.x + threadIdx.x;
		psiUCHAR[tid] = (unsigned char) ( -255*( values[field*nCells + tid]*normaliztion -1 ));
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__ void reduction_max_kernel( double *input, double *output ){
		__shared__ double sh_data[512];

		unsigned int tid = threadIdx.x;
		unsigned int i   = blockIdx.x * blockDim.x  + threadIdx.x;
		sh_data[tid] = max( input[i], input[i + blockDim.x*gridDim.x ] ) ;
		__syncthreads();

		for( unsigned int s = blockDim.x/2; s>0; s >>= 1){
			if ( tid < s ) sh_data[tid] = max( sh_data[tid], sh_data[tid+s] );
			__syncthreads();
		}

		if ( tid == 0 ) output[ blockIdx.x ] = sh_data[0];
	}
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__ void reduction_min_kernel( double *input, double *output ){
		__shared__ double sh_data[512];

		unsigned int tid = threadIdx.x;
		unsigned int i   = blockIdx.x * blockDim.x  + threadIdx.x;
		sh_data[tid] = min( input[i], input[i + blockDim.x*gridDim.x ] ) ;
		__syncthreads();

		for( unsigned int s = blockDim.x/2; s>0; s >>= 1){
			if ( tid < s ) sh_data[tid] = min( sh_data[tid], sh_data[tid+s] );
			__syncthreads();
		}

		if ( tid == 0 ) output[ blockIdx.x ] = sh_data[0];
	}
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

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	__device__ double hll_interFlux( double val_l, double val_r, double F_l, double F_r, double s_l, double s_r ){
	  if ( s_l > 0 ) return F_l;
	  if ( s_r < 0 ) return F_r;
	  return ( s_r*F_l - s_l*F_r + s_l*s_r*( val_r - val_l ) ) / ( s_r - s_l );
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	__device__ void writeInterFlux(const int coord, const int writeStride, int tid,
					double rho_l, double rho_r, double vx_l, double vx_r, double vy_l, double vy_r, double vz_l, double vz_r, double E_l, double E_r,
					double p_l, double p_r, double s_l, double s_r,
					double *iFlx ){

		// Adjacent fluxes from left and center cell
		double F_l, F_r;

		//iFlx rho
		if ( coord == 1 ){
			F_l = rho_l * vx_l;
			F_r = rho_r * vx_r;
		}
		else if ( coord == 2 ){
			F_l = rho_l * vy_l;
			F_r = rho_r * vy_r;
		}
		else if ( coord == 3 ){
			F_l = rho_l * vz_l;
			F_r = rho_r * vz_r;
		}
		iFlx[0*writeStride + tid] = hll_interFlux( rho_l, rho_r, F_l, F_r, s_l, s_r );

		//iFlx rho * vx
		if ( coord == 1 ){
			F_l = rho_l * vx_l * vx_l + p_l;
			F_r = rho_r * vx_r * vx_r + p_r;
		}
		else if ( coord == 2 ){
			F_l = rho_l * vx_l * vy_l;
			F_r = rho_r * vx_r * vy_r;
		}
		else if ( coord == 3 ){
			F_l = rho_l * vx_l * vz_l;
			F_r = rho_r * vx_r * vz_r;
		}
		iFlx[1*writeStride + tid] = hll_interFlux( rho_l*vx_l, rho_r*vx_r, F_l, F_r, s_l, s_r );

		//iFlx rho * vy
		if ( coord == 1 ){
			F_l = rho_l * vy_l * vx_l ;
			F_r = rho_r * vy_r * vx_r ;
		}
		else if ( coord == 2 ){
			F_l = rho_l * vy_l * vy_l + p_l;
			F_r = rho_r * vy_r * vy_r + p_r;
		}
		else if ( coord == 3 ){
			F_l = rho_l * vy_l * vz_l;
			F_r = rho_r * vy_r * vz_r;
		}
		iFlx[2*writeStride + tid] = hll_interFlux( rho_l*vy_l, rho_r*vy_r, F_l, F_r, s_l, s_r );

		//iFlx rho * vz
		if ( coord == 1 ){
			F_l = rho_l * vz_l * vx_l ;
			F_r = rho_r * vz_r * vx_r ;
		}
		else if ( coord == 2 ){
			F_l = rho_l * vz_l * vy_l ;
			F_r = rho_r * vz_r * vy_r ;
		}
		else if ( coord == 3 ){
			F_l = rho_l * vz_l * vz_l + p_l ;
			F_r = rho_r * vz_r * vz_r + p_r ;
		}
		iFlx[3*writeStride + tid] = hll_interFlux( rho_l*vz_l, rho_r*vz_r, F_l, F_r, s_l, s_r );

		//iFlx E
		if ( coord == 1 ){
			F_l = vx_l * ( E_l + p_l ) ;
			F_r = vx_r * ( E_r + p_r ) ;
		}
		else if ( coord == 2 ){
			F_l = vy_l * ( E_l + p_l ) ;
			F_r = vy_r * ( E_r + p_r ) ;
		}
		else if ( coord == 3 ){
			F_l = vz_l * ( E_l + p_l ) ;
			F_r = vz_r * ( E_r + p_r ) ;
		}
		iFlx[4*writeStride + tid] = hll_interFlux( E_l, E_r, F_l, F_r, s_l, s_r );
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  __global__ void setInterFlux_hll( const int coord, const int nCells, const double gamma, const double dx, const double dy, const double dz,
				 double *cnsv, double *iFlx,
				 double *bound_l, double *bound_r,
				 double *iFlx_bnd,
         double* times ){
    int t_j = blockIdx.x*blockDim.x + threadIdx.x;
    int t_i = blockIdx.y*blockDim.y + threadIdx.y;
    int t_k = blockIdx.z*blockDim.z + threadIdx.z;
    int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

    int tid_adj, boundId, stride;
    double v2;
    double rho_l, vx_l, vy_l, vz_l, E_l, p_l;
    double rho_c, vx_c, vy_c, vz_c, E_c, p_c;

    //Set adjacent id
    if ( coord == 1 ){
      if ( t_j == 0) tid_adj = (t_j) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
      else tid_adj = (t_j-1) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
    }
    if ( coord == 2 ){
      if ( t_i == 0) tid_adj = t_j + (t_i)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
      else tid_adj = t_j + (t_i-1)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
    }
    if ( coord == 3 ){
      if ( t_k == 0) tid_adj = t_j + t_i*blockDim.x*gridDim.x + (t_k)*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
      else tid_adj = t_j + t_i*blockDim.x*gridDim.x + (t_k-1)*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
    }
    //Read adjacent and center conservatives
    rho_l = cnsv[ 0*nCells + tid_adj ];
    rho_c = cnsv[ 0*nCells + tid ];

    vx_l = cnsv[ 1*nCells + tid_adj ] / rho_l;
    vx_c = cnsv[ 1*nCells + tid ] / rho_c;

    vy_l = cnsv[ 2*nCells + tid_adj ] / rho_l;
    vy_c = cnsv[ 2*nCells + tid ] / rho_c;

    vz_l = cnsv[ 3*nCells + tid_adj ] / rho_l;
    vz_c = cnsv[ 3*nCells + tid ] / rho_c;

    E_l = cnsv[ 4*nCells + tid_adj ];
    E_c = cnsv[ 4*nCells + tid ];

    //Load and apply boundery conditions
    if ( coord == 1 ){
      boundId = t_i + t_k*N_H;
			stride = N_H*N_D;
      if ( t_j == 0) {
        rho_l = bound_l[0*stride + boundId];
        vx_l  = bound_l[1*stride + boundId] / rho_l;
        vy_l  = bound_l[2*stride + boundId] / rho_l;
        vz_l  = bound_l[3*stride + boundId] / rho_l;
        E_l   = bound_l[4*stride + boundId];
      }
    }
    if ( coord == 2 ){
      boundId = t_j + t_k*N_W;
			stride = N_W*N_D;
      if ( t_i == 0) {
        rho_l = bound_l[0*stride + boundId];
        vx_l  = bound_l[1*stride + boundId] / rho_l;
        vy_l  = bound_l[2*stride + boundId] / rho_l;
        vz_l  = bound_l[3*stride + boundId] / rho_l;
        E_l   = bound_l[4*stride + boundId];
      }
    }
    if ( coord == 3 ){
      boundId = t_j + t_i*N_W;
			stride = N_W*N_H;
      if ( t_k == 0) {
        rho_l = bound_l[0*stride + boundId];
        vx_l  = bound_l[1*stride + boundId] / rho_l;
        vy_l  = bound_l[2*stride + boundId] / rho_l;
        vz_l  = bound_l[3*stride + boundId] / rho_l;
        E_l   = bound_l[4*stride + boundId];
      }
    }


    //Boundary bounce condition
      if ( coord==1 && t_j == 0 ) vx_l = -vx_c;
        //Boundary bounce condition
      if ( coord==2 && t_i == 0 ) vy_l = -vy_c;
      //Boundary bounce condition
      if ( coord==3 && t_k == 0 ) vz_l = -vz_c;

    v2    = vx_l*vx_l + vy_l*vy_l + vz_l*vz_l;
    p_l   = ( E_l - rho_l*v2/2 ) * (gamma-1);

    v2    = vx_c*vx_c + vy_c*vy_c + vz_c*vz_c;
    p_c   = ( E_c - rho_c*v2/2 ) * (gamma-1);

    double cs_l, cs_c, s_l, s_c;
    cs_l = sqrt( p_l * gamma / rho_l );
    cs_c = sqrt( p_c * gamma / rho_c );

    if ( coord == 1 ){
      s_l = min( vx_l - cs_l, vx_c - cs_c );
      s_c = max( vx_l + cs_l, vx_c + cs_c );
      //Use v2 to save time minimum
      v2 = dx / ( abs( vx_c ) + cs_c );
      v2 = min( v2, dy / ( abs( vy_c ) + cs_c ) );
      v2 = min( v2, dz / ( abs( vz_c ) + cs_c ) );
      times[ tid ] = v2;
    }

    else if ( coord == 2 ){
      s_l = min( vy_l - cs_l, vy_c - cs_c );
      s_c = max( vy_l + cs_l, vy_c + cs_c ); 
    }

    else if ( coord == 3 ){
      s_l = min( vz_l - cs_l, vz_c - cs_c );
      s_c = max( vz_l + cs_l, vz_c + cs_c );
    }

    writeInterFlux( coord, nCells, tid, rho_l, rho_c, vx_l, vx_c, vy_l, vy_c, vz_l, vz_c, E_l, E_c,
            p_l, p_c, s_l, s_c, iFlx  );

    //Get iFlux_r for most right cell
    // if ( blockIdx.x!=(gridDim.x-1) || blockIdx.y!=(gridDim.y-1) || blockIdx.z!=(gridDim.z-1) ) return;

    if ( coord == 1 ){
      if ( t_j != (N_W-1) ) return;
    }
    if ( coord == 2 ){
      if ( t_i != (N_H-1) ) return;
    }
    if ( coord == 3 ){
      if ( t_k != (N_D-1) ) return;
    }

    rho_l = rho_c;
    vx_l = vx_c;
    vy_l = vy_c;
    vz_l = vz_c;
    E_l = E_c;
    p_l = p_c;
    cs_l = cs_c;



    //Load Bounderies for right part of the box_size
    rho_c = bound_r[0*stride + boundId];
    vx_c  = bound_r[1*stride + boundId] / rho_c;
    vy_c  = bound_r[2*stride + boundId] / rho_c;
    vz_c  = bound_r[3*stride + boundId] / rho_c;
    E_c   = bound_r[4*stride + boundId];

    // //Boundary bounce conditions
    if ( coord==1 && t_j == (N_W-1) ) vx_c = -vx_l;
      //Boundary bounce condition
    if ( coord==2 && t_i == (N_H-1) ) vy_c = -vy_l;
    //Boundary bounce condition
    if ( coord==3 && t_k == (N_D-1) ) vz_c = -vz_l;

    v2    = vx_c*vx_c + vy_c*vy_c + vz_c*vz_c;
    p_c   = ( E_c - rho_c*v2/2 ) * (gamma-1);
    cs_c = sqrt( p_c * gamma / rho_c );
    if ( coord == 1 ){
      s_l = min( vx_l - cs_l, vx_c - cs_c );
      s_c = max( vx_l + cs_l, vx_c + cs_c );
    }

    else if ( coord == 2 ){
      s_l = min( vy_l - cs_l, vy_c - cs_c );
      s_c = max( vy_l + cs_l, vy_c + cs_c );
    }

    else if ( coord == 3 ){
      s_l = min( vz_l - cs_l, vz_c - cs_c );
      s_c = max( vz_l + cs_l, vz_c + cs_c );
    }
		writeInterFlux( coord, stride, boundId, rho_l, rho_c, vx_l, vx_c, vy_l, vy_c, vz_l, vz_c, E_l, E_c,
            p_l, p_c, s_l, s_c, iFlx_bnd  );
    // writeInterFlux_b( coord, boundId, rho_l, rho_c, vx_l, vx_c, vy_l, vy_c, vz_l, vz_c, E_l, E_c,
    //         p_l, p_c, s_l, s_c, iFlx, iFlx_1_bnd, iFlx_2_bnd, iFlx_3_bnd, iFlx_4_bnd, iFlx_5_bnd  );
  }
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	__global__ void getInterFlux_hll( const int coord, const int nCells, const double dt,  const double gamma,
				 const double dx, const double dy, const double dz,
				 double *cnsv_adv, double *iFlx, double *iFlx_bnd ){
				 //  double* gForceX, double* gForceY, double* gForceZ, double* gravWork ){
		int t_j = blockIdx.x*blockDim.x + threadIdx.x;
		int t_i = blockIdx.y*blockDim.y + threadIdx.y;
		int t_k = blockIdx.z*blockDim.z + threadIdx.z;
		int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

		int tid_adj, boundId, stride;
		double iFlx1_l, iFlx2_l, iFlx3_l, iFlx4_l, iFlx5_l;
		double iFlx1_r, iFlx2_r, iFlx3_r, iFlx4_r, iFlx5_r;
		double delta;

		//Set adjacent id
		if ( coord == 1 ){
			if ( t_j == N_W-1 ) tid_adj = (t_j) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
			else tid_adj = (t_j+1) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
			delta = dt / dx;
		}
		if ( coord == 2 ){
			if ( t_i == N_H-1 ) tid_adj = t_j + (t_i)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
			else tid_adj = t_j + (t_i+1)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
			delta = dt / dy;
		}
		if ( coord == 3 ){
			if ( t_k == N_D-1) tid_adj = t_j + t_i*blockDim.x*gridDim.x + (t_k)*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
			else tid_adj = t_j + t_i*blockDim.x*gridDim.x + (t_k+1)*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
			delta = dt / dz;
		}

		//Read inter-cell fluxes
		iFlx1_l = iFlx[0*nCells + tid ];
		iFlx1_r = iFlx[0*nCells + tid_adj ];

		iFlx2_l = iFlx[1*nCells + tid ];
		iFlx2_r = iFlx[1*nCells + tid_adj ];

		iFlx3_l = iFlx[2*nCells + tid ];
		iFlx3_r = iFlx[2*nCells + tid_adj ];

		iFlx4_l = iFlx[3*nCells + tid ];
		iFlx4_r = iFlx[3*nCells + tid_adj ];

		iFlx5_l = iFlx[4*nCells + tid ];
		iFlx5_r = iFlx[4*nCells + tid_adj ];

		if ( coord == 1 ){
			boundId = t_i + t_k*N_H;
			stride = N_H*N_D;
			if ( t_j == (N_W-1) ) {
				iFlx1_r = iFlx_bnd[0*stride + boundId];
				iFlx2_r = iFlx_bnd[1*stride + boundId];
				iFlx3_r = iFlx_bnd[2*stride + boundId];
				iFlx4_r = iFlx_bnd[3*stride + boundId];
				iFlx5_r = iFlx_bnd[4*stride + boundId];
			}
		}
		if ( coord == 2 ){
			boundId = t_j + t_k*N_W;
			stride = N_W*N_D;
			if ( t_i == (N_H-1) ) {
				iFlx1_r = iFlx_bnd[0*stride + boundId];
				iFlx2_r = iFlx_bnd[1*stride + boundId];
				iFlx3_r = iFlx_bnd[2*stride + boundId];
				iFlx4_r = iFlx_bnd[3*stride + boundId];
				iFlx5_r = iFlx_bnd[4*stride + boundId];
			}
		}
		if ( coord == 3 ){
			boundId = t_j + t_i*N_W;
			stride = N_W*N_H;
			if ( t_k == (N_D-1) ) {
				iFlx1_r = iFlx_bnd[0*stride + boundId];
				iFlx2_r = iFlx_bnd[1*stride + boundId];
				iFlx3_r = iFlx_bnd[2*stride + boundId];
				iFlx4_r = iFlx_bnd[3*stride + boundId];
				iFlx5_r = iFlx_bnd[4*stride + boundId];
			}
		}
		//Load and apply boundery conditions
		// if ( coord == 1 ){
		//   boundId = t_i + t_k*N_H;
		//   if ( t_j == (N_W-1) ) {
		//     iFlx1_r = iFlx_1_bnd[boundId];
		//     iFlx2_r = iFlx_2_bnd[boundId];
		//     iFlx3_r = iFlx_3_bnd[boundId];
		//     iFlx4_r = iFlx_4_bnd[boundId];
		//     iFlx5_r = iFlx_5_bnd[boundId];
		//   }
		// }
		// if ( coord == 2 ){
		//   boundId = t_j + t_k*N_W;
		//   if ( t_i == (N_H-1) ) {
		//     iFlx1_r = iFlx_1_bnd[boundId];
		//     iFlx2_r = iFlx_2_bnd[boundId];
		//     iFlx3_r = iFlx_3_bnd[boundId];
		//     iFlx4_r = iFlx_4_bnd[boundId];
		//     iFlx5_r = iFlx_5_bnd[boundId];
		//   }
		// }
		// if ( coord == 3 ){
		//   boundId = t_j + t_i*N_W;
		//   if ( t_k == (N_D-1) ) {
		//     iFlx1_r = iFlx_1_bnd[boundId];
		//     iFlx2_r = iFlx_2_bnd[boundId];
		//     iFlx3_r = iFlx_3_bnd[boundId];
		//     iFlx4_r = iFlx_4_bnd[boundId];
		//     iFlx5_r = iFlx_5_bnd[boundId];
		//   }
		// }

		//Advance the consv values
		// cnsv_1[ tid ] = cnsv_1[ tid ] - delta*( iFlx1_r - iFlx1_l );
		// cnsv_2[ tid ] = cnsv_2[ tid ] - delta*( iFlx2_r - iFlx2_l ) + dt*gForceX[tid]*50;
		// cnsv_3[ tid ] = cnsv_3[ tid ] - delta*( iFlx3_r - iFlx3_l ) + dt*gForceY[tid]*50;
		// cnsv_4[ tid ] = cnsv_4[ tid ] - delta*( iFlx4_r - iFlx4_l ) + dt*gForceZ[tid]*50;
		// cnsv_5[ tid ] = cnsv_5[ tid ] - delta*( iFlx5_r - iFlx5_l ) + dt*gravWork[tid]*50;

		if ( coord == 1 ){
			cnsv_adv[0*nCells +  tid ] = -delta*( iFlx1_r - iFlx1_l );
			cnsv_adv[1*nCells +  tid ] = -delta*( iFlx2_r - iFlx2_l );
			cnsv_adv[2*nCells +  tid ] = -delta*( iFlx3_r - iFlx3_l );
			cnsv_adv[3*nCells +  tid ] = -delta*( iFlx4_r - iFlx4_l );
			cnsv_adv[4*nCells +  tid ] = -delta*( iFlx5_r - iFlx5_l );
		}
		else{
			cnsv_adv[0*nCells +  tid ] = cnsv_adv[0*nCells +  tid ] - delta*( iFlx1_r - iFlx1_l );
			cnsv_adv[1*nCells +  tid ] = cnsv_adv[1*nCells +  tid ] - delta*( iFlx2_r - iFlx2_l );
			cnsv_adv[2*nCells +  tid ] = cnsv_adv[2*nCells +  tid ] - delta*( iFlx3_r - iFlx3_l );
			cnsv_adv[3*nCells +  tid ] = cnsv_adv[3*nCells +  tid ] - delta*( iFlx4_r - iFlx4_l );
			cnsv_adv[4*nCells +  tid ] = cnsv_adv[4*nCells +  tid ] - delta*( iFlx5_r - iFlx5_l );
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	__global__ void addDtoD( const int nCells,
			double *dst, double *sum ){
		int t_j = blockIdx.x*blockDim.x + threadIdx.x;
		int t_i = blockIdx.y*blockDim.y + threadIdx.y;
		int t_k = blockIdx.z*blockDim.z + threadIdx.z;
		int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

		dst[0*nCells + tid] = dst[0*nCells + tid] + sum[0*nCells + tid];
		dst[1*nCells + tid] = dst[1*nCells + tid] + sum[1*nCells + tid];
		dst[2*nCells + tid] = dst[2*nCells + tid] + sum[2*nCells + tid];
		dst[3*nCells + tid] = dst[3*nCells + tid] + sum[3*nCells + tid];
		dst[4*nCells + tid] = dst[4*nCells + tid] + sum[4*nCells + tid];
	}
}//End of extern 'C'

















//
// //Textures for conserv
// texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_1;
// texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_2;
// texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_3;
// texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_4;
// texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_5;
//
//
// //Surfaces for Fluxes
// surface< void, cudaSurfaceType3D> surf_1;
// surface< void, cudaSurfaceType3D> surf_2;
// surface< void, cudaSurfaceType3D> surf_3;
// surface< void, cudaSurfaceType3D> surf_4;
// surface< void, cudaSurfaceType3D> surf_5;
//
// __global__ void setInterFlux_hll( const int coord, const cudaP gamma, const cudaP dx, const cudaP dy, const cudaP dz,
// 			 cudaP* cnsv_1, cudaP* cnsv_2, cudaP* cnsv_3, cudaP* cnsv_4, cudaP* cnsv_5,
// 			 float* times ){
//   int t_j = blockIdx.x*blockDim.x + threadIdx.x;
//   int t_i = blockIdx.y*blockDim.y + threadIdx.y;
//   int t_k = blockIdx.z*blockDim.z + threadIdx.z;
//   int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//
//   cudaP v2;
//   cudaP rho_l, vx_l, vy_l, vz_l, E_l, p_l;
//   cudaP rho_c, vx_c, vy_c, vz_c, E_c, p_c;
// //   float time;
//   //Read adjacent conserv
//   if ( coord == 1 ){
//     rho_l = fp_tex3D( tex_1, t_j-1, t_i, t_k);
//     rho_c = fp_tex3D( tex_1, t_j, t_i, t_k);
//
//     vx_l  = fp_tex3D( tex_2, t_j-1, t_i, t_k) / rho_l;
//     vx_c  = fp_tex3D( tex_2, t_j, t_i, t_k)   / rho_c;
//
//     vy_l  = fp_tex3D( tex_3, t_j-1, t_i, t_k) / rho_l;
//     vy_c  = fp_tex3D( tex_3, t_j, t_i, t_k)   / rho_c;
//
//     vz_l  = fp_tex3D( tex_4, t_j-1, t_i, t_k) / rho_l;
//     vz_c  = fp_tex3D( tex_4, t_j, t_i, t_k)   / rho_c;
//
//     E_l   = fp_tex3D( tex_5, t_j-1, t_i, t_k);
//     E_c   = fp_tex3D( tex_5, t_j, t_i, t_k);
//
//
//   }
//   else if ( coord == 2 ){
//     rho_l = fp_tex3D( tex_1, t_j, t_i-1, t_k);
//     rho_c = fp_tex3D( tex_1, t_j, t_i, t_k);
//
//     vx_l  = fp_tex3D( tex_2, t_j, t_i-1, t_k) / rho_l;
//     vx_c  = fp_tex3D( tex_2, t_j, t_i, t_k)   / rho_c;
//
//     vy_l  = fp_tex3D( tex_3, t_j, t_i-1, t_k) / rho_l;
//     vy_c  = fp_tex3D( tex_3, t_j, t_i, t_k)   / rho_c;
//
//     vz_l  = fp_tex3D( tex_4, t_j, t_i-1, t_k) / rho_l;
//     vz_c  = fp_tex3D( tex_4, t_j, t_i, t_k)   / rho_c;
//
//     E_l   = fp_tex3D( tex_5, t_j, t_i-1, t_k);
//     E_c   = fp_tex3D( tex_5, t_j, t_i, t_k);
//
//
//   }
//   else if ( coord == 3 ){
//     rho_l = fp_tex3D( tex_1, t_j, t_i, t_k-1);
//     rho_c = fp_tex3D( tex_1, t_j, t_i, t_k);
//
//     vx_l  = fp_tex3D( tex_2, t_j, t_i, t_k-1) / rho_l;
//     vx_c  = fp_tex3D( tex_2, t_j, t_i, t_k)   / rho_c;
//
//     vy_l  = fp_tex3D( tex_3, t_j, t_i, t_k-1) / rho_l;
//     vy_c  = fp_tex3D( tex_3, t_j, t_i, t_k)   / rho_c;
//
//     vz_l  = fp_tex3D( tex_4, t_j, t_i, t_k-1) / rho_l;
//     vz_c  = fp_tex3D( tex_4, t_j, t_i, t_k)   / rho_c;
//
//     E_l   = fp_tex3D( tex_5, t_j, t_i, t_k-1);
//     E_c   = fp_tex3D( tex_5, t_j, t_i, t_k);
//
//
//   }
// //   //Boundary bounce condition
// //     if ( t_j == 0 ) vx_l = -vx_c;
// //       //Boundary bounce condition
// //     if ( t_i == 0 ) vy_l = -vy_c;
// //     //Boundary bounce condition
// //     if ( t_k == 0 ) vz_l = -vz_c;
//
//   v2    = vx_l*vx_l + vy_l*vy_l + vz_l*vz_l;
//   p_l   = ( E_l - rho_l*v2/2 ) * (gamma-1);
//
//   v2    = vx_c*vx_c + vy_c*vy_c + vz_c*vz_c;
//   p_c   = ( E_c - rho_c*v2/2 ) * (gamma-1);
//
//
//   cudaP cs_l, cs_c, s_l, s_c;
//   cs_l = sqrt( p_l * gamma / rho_l );
//   cs_c = sqrt( p_c * gamma / rho_c );
//
//
//
//   if ( coord == 1 ){
//     s_l = min( vx_l - cs_l, vx_c - cs_c );
//     s_c = max( vx_l + cs_l, vx_c + cs_c );
//     //Use v2 to save time minimum
//     v2 = dx / ( abs( vx_c ) + cs_c );
//     v2 = min( v2, dy / ( abs( vy_c ) + cs_c ) );
//     v2 = min( v2, dz / ( abs( vz_c ) + cs_c ) );
//     times[ tid ] = v2;
//   }
//   else if ( coord == 2 ){
//     s_l = min( vy_l - cs_l, vy_c - cs_c );
//     s_c = max( vy_l + cs_l, vy_c + cs_c );
//   }
//   else if ( coord == 3 ){
//     s_l = min( vz_l - cs_l, vz_c - cs_c );
//     s_c = max( vz_l + cs_l, vz_c + cs_c );
//   }
//
//   // Adjacent fluxes from left and center cell
//   cudaP F_l, F_c, iFlx;
//
//   //iFlx rho
//   if ( coord == 1 ){
//     F_l = rho_l * vx_l;
//     F_c = rho_c * vx_c;
//   }
//   else if ( coord == 2 ){
//     F_l = rho_l * vy_l;
//     F_c = rho_c * vy_c;
//   }
//   else if ( coord == 3 ){
//     F_l = rho_l * vz_l;
//     F_c = rho_c * vz_c;
//   }
//   if ( s_l > 0 ) iFlx = F_l;
//   else if ( s_c < 0 ) iFlx = F_c;
//   else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( rho_c - rho_l ) ) / ( s_c - s_l );
//   surf3Dwrite(  iFlx, surf_1,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
//
//   //iFlx rho * vx
//   if ( coord == 1 ){
//     F_l = rho_l * vx_l * vx_l + p_l;
//     F_c = rho_c * vx_c * vx_c + p_c;
//   }
//   else if ( coord == 2 ){
//     F_l = rho_l * vx_l * vy_l;
//     F_c = rho_c * vx_c * vy_c;
//   }
//   else if ( coord == 3 ){
//     F_l = rho_l * vx_l * vz_l;
//     F_c = rho_c * vx_c * vz_c;
//   }
//   if ( s_l > 0 ) iFlx = F_l;
//   else if ( s_c < 0 ) iFlx = F_c;
//   else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( rho_c*vx_c - rho_l*vx_l ) ) / ( s_c - s_l );
//   surf3Dwrite(  iFlx, surf_2,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
//
//   //iFlx rho * vy
//   if ( coord == 1 ){
//     F_l = rho_l * vy_l * vx_l ;
//     F_c = rho_c * vy_c * vx_c ;
//   }
//   else if ( coord == 2 ){
//     F_l = rho_l * vy_l * vy_l + p_l;
//     F_c = rho_c * vy_c * vy_c + p_c;
//   }
//   else if ( coord == 3 ){
//     F_l = rho_l * vy_l * vz_l;
//     F_c = rho_c * vy_c * vz_c;
//   }
//   if ( s_l > 0 ) iFlx = F_l;
//   else if ( s_c < 0 ) iFlx = F_c;
//   else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( rho_c*vy_c - rho_l*vy_l ) ) / ( s_c - s_l );
//   surf3Dwrite(  iFlx, surf_3,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
//
//   //iFlx rho * vz
//   if ( coord == 1 ){
//     F_l = rho_l * vz_l * vx_l ;
//     F_c = rho_c * vz_c * vx_c ;
//   }
//   else if ( coord == 2 ){
//     F_l = rho_l * vz_l * vy_l ;
//     F_c = rho_c * vz_c * vy_c ;
//   }
//   else if ( coord == 3 ){
//     F_l = rho_l * vz_l * vz_l + p_l ;
//     F_c = rho_c * vz_c * vz_c + p_c ;
//   }
//   if ( s_l > 0 ) iFlx = F_l;
//   else if ( s_c < 0 ) iFlx = F_c;
//   else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( rho_c*vz_c - rho_l*vz_l ) ) / ( s_c - s_l );
//   surf3Dwrite(  iFlx, surf_4,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
//
//   //iFlx E
//   if ( coord == 1 ){
//     F_l = vx_l * ( E_l + p_l ) ;
//     F_c = vx_c * ( E_c + p_c ) ;
//   }
//   else if ( coord == 2 ){
//     F_l = vy_l * ( E_l + p_l ) ;
//     F_c = vy_c * ( E_c + p_c ) ;
//   }
//   else if ( coord == 3 ){
//     F_l = vz_l * ( E_l + p_l ) ;
//     F_c = vz_c * ( E_c + p_c ) ;
//   }
//   if ( s_l > 0 ) iFlx = F_l;
//   else if ( s_c < 0 ) iFlx = F_c;
//   else  iFlx = ( s_c*F_l - s_l*F_c + s_l*s_c*( E_c - E_l ) ) / ( s_c - s_l );
//   surf3Dwrite(  iFlx, surf_5,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
//
// }
//
// __global__ void getInterFlux_hll( const int coord, const cudaP dt,  const cudaP gamma, const cudaP dx, const cudaP dy, const cudaP dz,
// 			 cudaP* cnsv_1, cudaP* cnsv_2, cudaP* cnsv_3, cudaP* cnsv_4, cudaP* cnsv_5,
// 			 cudaP* gForceX, cudaP* gForceY, cudaP* gForceZ, cudaP* gravWork ){
//   int t_j = blockIdx.x*blockDim.x + threadIdx.x;
//   int t_i = blockIdx.y*blockDim.y + threadIdx.y;
//   int t_k = blockIdx.z*blockDim.z + threadIdx.z;
//   int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//
//   //Read inter-cell fluxes from textures
//
//   cudaP iFlx1_l, iFlx2_l, iFlx3_l, iFlx4_l, iFlx5_l;
//   cudaP iFlx1_r, iFlx2_r, iFlx3_r, iFlx4_r, iFlx5_r;
//   cudaP delta;
//   if ( coord == 1 ){
//     delta = dt / dx;
//     iFlx1_l = fp_tex3D( tex_1, t_j, t_i, t_k);
//     iFlx1_r = fp_tex3D( tex_1, t_j+1, t_i, t_k);
//
//     iFlx2_l = fp_tex3D( tex_2, t_j, t_i, t_k);
//     iFlx2_r = fp_tex3D( tex_2, t_j+1, t_i, t_k);
//
//     iFlx3_l = fp_tex3D( tex_3, t_j, t_i, t_k);
//     iFlx3_r = fp_tex3D( tex_3, t_j+1, t_i, t_k);
//
//     iFlx4_l = fp_tex3D( tex_4, t_j, t_i, t_k);
//     iFlx4_r = fp_tex3D( tex_4, t_j+1, t_i, t_k);
//
//     iFlx5_l = fp_tex3D( tex_5, t_j, t_i, t_k);
//     iFlx5_r = fp_tex3D( tex_5, t_j+1, t_i, t_k);
//   }
//   else if ( coord == 2 ){
//     delta = dt / dy;
//     iFlx1_l = fp_tex3D( tex_1, t_j, t_i, t_k);
//     iFlx1_r = fp_tex3D( tex_1, t_j, t_i+1, t_k);
//
//     iFlx2_l = fp_tex3D( tex_2, t_j, t_i, t_k);
//     iFlx2_r = fp_tex3D( tex_2, t_j, t_i+1, t_k);
//
//     iFlx3_l = fp_tex3D( tex_3, t_j, t_i, t_k);
//     iFlx3_r = fp_tex3D( tex_3, t_j, t_i+1, t_k);
//
//     iFlx4_l = fp_tex3D( tex_4, t_j, t_i, t_k);
//     iFlx4_r = fp_tex3D( tex_4, t_j, t_i+1, t_k);
//
//     iFlx5_l = fp_tex3D( tex_5, t_j, t_i, t_k);
//     iFlx5_r = fp_tex3D( tex_5, t_j, t_i+1, t_k);
//   }
//   else if ( coord == 3 ){
//     delta = dt / dz;
//     iFlx1_l = fp_tex3D( tex_1, t_j, t_i, t_k);
//     iFlx1_r = fp_tex3D( tex_1, t_j, t_i, t_k+1);
//
//     iFlx2_l = fp_tex3D( tex_2, t_j, t_i, t_k);
//     iFlx2_r = fp_tex3D( tex_2, t_j, t_i, t_k+1);
//
//     iFlx3_l = fp_tex3D( tex_3, t_j, t_i, t_k);
//     iFlx3_r = fp_tex3D( tex_3, t_j, t_i, t_k+1);
//
//     iFlx4_l = fp_tex3D( tex_4, t_j, t_i, t_k);
//     iFlx4_r = fp_tex3D( tex_4, t_j, t_i, t_k+1);
//
//     iFlx5_l = fp_tex3D( tex_5, t_j, t_i, t_k);
//     iFlx5_r = fp_tex3D( tex_5, t_j, t_i, t_k+1);
//   }
//
//   //Advance the consv values
//   // cnsv_1[ tid ] = cnsv_1[ tid ] - delta*( iFlx1_r - iFlx1_l );
//   // cnsv_2[ tid ] = cnsv_2[ tid ] - delta*( iFlx2_r - iFlx2_l ) + dt*gForceX[tid]*50;
//   // cnsv_3[ tid ] = cnsv_3[ tid ] - delta*( iFlx3_r - iFlx3_l ) + dt*gForceY[tid]*50;
//   // cnsv_4[ tid ] = cnsv_4[ tid ] - delta*( iFlx4_r - iFlx4_l ) + dt*gForceZ[tid]*50;
//   // cnsv_5[ tid ] = cnsv_5[ tid ] - delta*( iFlx5_r - iFlx5_l ) + dt*gravWork[tid]*50;
//
//   cnsv_1[ tid ] = cnsv_1[ tid ] - delta*( iFlx1_r - iFlx1_l );
//   cnsv_2[ tid ] = cnsv_2[ tid ] - delta*( iFlx2_r - iFlx2_l );
//   cnsv_3[ tid ] = cnsv_3[ tid ] - delta*( iFlx3_r - iFlx3_l );
//   cnsv_4[ tid ] = cnsv_4[ tid ] - delta*( iFlx4_r - iFlx4_l );
//   cnsv_5[ tid ] = cnsv_5[ tid ] - delta*( iFlx5_r - iFlx5_l );
// }
//
//
// __global__ void iterPoissonStep( int* converged, const int paridad,
// 				 const int nWidth, const cudaP omega, const cudaP pi4,
// 				 cudaP dx, cudaP dy, cudaP dz,
// 				 cudaP* rhoVals, cudaP* phiVals, float* phiWall ){
//   int t_j = 2*(blockIdx.x*blockDim.x + threadIdx.x);
//   int t_i = blockIdx.y*blockDim.y + threadIdx.y;
//   int t_k = blockIdx.z*blockDim.z + threadIdx.z;
//   //Make a checkboard 3D grid
//   if ( t_i%2 == 0 ){
//     if ( t_k%2 == paridad ) t_j +=1;
//   }
//   else if ( (t_k+1)%2 == paridad ) t_j +=1;
//   int tid = t_j + t_i*nWidth + t_k*nWidth*blockDim.y*gridDim.y;
//
//   cudaP rho, phi_c, phi_l, phi_r, phi_d, phi_u, phi_b, phi_t, phi_new;
//   rho = rhoVals[ tid ];
//   phi_c = fp_tex3D( tex_1, t_j, t_i, t_k);
//   phi_l = fp_tex3D( tex_1, t_j-1, t_i, t_k);
//   phi_r = fp_tex3D( tex_1, t_j+1, t_i, t_k);
//   phi_d = fp_tex3D( tex_1, t_j, t_i-1, t_k);
//   phi_u = fp_tex3D( tex_1, t_j, t_i+1, t_k);
//   phi_b = fp_tex3D( tex_1, t_j, t_i, t_k-1);
//   phi_t = fp_tex3D( tex_1, t_j, t_i, t_k+1);
//
//   //Boundary conditions
//   if  ( t_j == 0 )        phi_l = phi_r;
//   if  ( t_j == nWidth-1 ) phi_r = phi_l;
//   if  ( t_i == 0 )        phi_d = phi_u;
//   if  ( t_i == nWidth-1 ) phi_u = phi_d;
//   if  ( t_k == 0 )        phi_b = phi_t;
//   if  ( t_k == nWidth-1 ) phi_t = phi_b;
//
// //   phi_new =  1./6 * ( phi_l + phi_r + phi_d + phi_u + phi_b + phi_t - dx*dx*rho   );
//   phi_new = (1-omega)*phi_c + omega/6*( phi_l + phi_r + phi_d + phi_u + phi_b + phi_t - dx*dx*pi4*rho );
//
//   if ( paridad == 0 ) surf3Dwrite(  phi_new, surf_1,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
//   phiVals[ tid ] = phi_new;
//
// //   if ( ( t_j == 0 ) ||  ( t_j == nWidth-1 ) || ( t_i == 0 ) || ( t_i == nWidth-1 ) || ( t_k == 0 ) || ( t_k == nWidth-1 ) ) return;
// //   if ( ( blockIdx.x == 0 ) ||  ( blo == nWidth-1 ) || ( t_i == 0 ) || ( t_i == nWidth-1 ) || ( t_k == 0 ) || ( t_k == nWidth-1 ) ) return;
//
//   if ( ( abs( ( phi_new - phi_c ) / phi_c ) > 0.002 ) ) converged[0] = 0;
//
//
// }
//
// __global__ void getGravityForce( const int nWidth, const int nHeight, const int nDepth,
// 				 cudaP dx, cudaP dy, cudaP dz,
// 				 cudaP* gForce_x, cudaP* gForce_y, cudaP* gForce_z,
// 				 cudaP* rho, cudaP* pX, cudaP* pY, cudaP* pZ, cudaP *gravWork,
// 				 float* phiWall      ){
//   int t_j = blockIdx.x*blockDim.x + threadIdx.x;
//   int t_i = blockIdx.y*blockDim.y + threadIdx.y;
//   int t_k = blockIdx.z*blockDim.z + threadIdx.z;
//   int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//
//   cudaP phi_l, phi_r, phi_d, phi_u, phi_b, phi_t;
// //   phi_c = fp_tex3D( tex_1, t_j, t_i, t_k);
//   phi_l = fp_tex3D( tex_1, t_j-1, t_i, t_k);
//   phi_r = fp_tex3D( tex_1, t_j+1, t_i, t_k);
//   phi_d = fp_tex3D( tex_1, t_j, t_i-1, t_k);
//   phi_u = fp_tex3D( tex_1, t_j, t_i+1, t_k);
//   phi_b = fp_tex3D( tex_1, t_j, t_i, t_k-1);
//   phi_t = fp_tex3D( tex_1, t_j, t_i, t_k+1);
//
//   //Boundary conditions
//   if  ( t_j == 0 )        phi_l = phi_r;
//   if  ( t_j == nWidth-1 ) phi_r = phi_l;
//   if  ( t_i == 0 )        phi_d = phi_u;
//   if  ( t_i == nWidth-1 ) phi_u = phi_d;
//   if  ( t_k == 0 )        phi_b = phi_t;
//   if  ( t_k == nWidth-1 ) phi_t = phi_b;
//
//   //Get partial derivatives for force
//   cudaP gField_x, gField_y, gField_z, p_x, p_y, p_z, rho_c;
//   rho_c = rho[ tid ];
//   gField_x = ( phi_l - phi_r ) * 0.5 / dx;
//   gField_y = ( phi_d - phi_u ) * 0.5 / dy;
//   gField_z = ( phi_b - phi_t ) * 0.5 / dz;
//   gForce_x[ tid ] = gField_x * rho_c;
//   gForce_y[ tid ] = gField_y * rho_c;
//   gForce_z[ tid ] = gField_z * rho_c;
// //   gForce_x[ tid ] = gField_x;
// //   gForce_y[ tid ] = gField_y;
// //   gForce_z[ tid ] = gField_z;
//
//   //Get momentum for virtual gravitational work
//   p_x = pX[ tid ] ;
//   p_y = pY[ tid ] ;
//   p_z = pZ[ tid ] ;
//   gravWork[ tid ] = p_x * gField_x + p_y * gField_y + p_z * gField_z ;
//
// }
//
// __global__ void reduceDensity( const int nWidth, const int nHeight, const int nDepth,
// 			       const float dx, const float dy, const float dz,
// 			       const float xMin, const float yMin, const float zMin,
// 			       cudaP* rhoAll, float* rhoReduced,
// 			       float* blockX, float* blockY, float* blockZ  ){
//   int t_j = blockIdx.x*blockDim.x + threadIdx.x;
//   int t_i = blockIdx.y*blockDim.y + threadIdx.y;
//   int t_k = blockIdx.z*blockDim.z + threadIdx.z;
//   int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
//   int tid_b = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
//   int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
//
//   float rho = float( rhoAll[tid] );
//   __shared__ float density[ THREADS_PER_BLOCK ];
//
//   density[tid_b] = rho;
//   __syncthreads();
//
//   int i =  blockDim.x * blockDim.y * blockDim.z  / 2;
//   while ( i > 0 ){
//     if ( tid_b < i ) density[tid_b] = density[tid_b] + density[tid_b+i];
//     __syncthreads();
//     i /= 2;
//   }
//
//
//   float x = blockDim.x*dx * ( blockIdx.x + 0.5f ) + xMin;
//   float y = blockDim.y*dy * ( blockIdx.y + 0.5f ) + yMin;
//   float z = blockDim.z*dz * ( blockIdx.z + 0.5f ) + zMin;
//   if (tid_b == 0 ){
//     rhoReduced[ bid ] = density[0]*dx*dy*dz ;
//     blockX[ bid ] = x;
//     blockY[ bid ] = y;
//     blockZ[ bid ] = z;
//   }
//
// }
// __global__ void getBounderyPotential(const float pi4, const int nBlocks, const int nWidth, const int nHeight, const int nDepth,
// 			      float dx, float dy, float dz, float xMin, float yMin, float zMin,
// 			      float* rhoReduced,  float* phiWall,
// 			      float* blockX, float* blockY, float* blockZ   ){
// // 			      float* phiWall_l, float* phiWall_r, float* phiWall_d, float* phiWall_u, float* phiWall_b, float* phiWall_t){
//   int t_j = blockIdx.x*blockDim.x + threadIdx.x;
//   int t_i = blockIdx.y*blockDim.y + threadIdx.y;
//   int tid = t_j + t_i*blockDim.x*gridDim.x ;
// //   int tid_b = threadIdx.x + threadIdx.y*blockDim.x
// //   int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
//
//   float y_wall = yMin + t_j*dy;
//   float z_wall = zMin + t_i*dz;
//
//   float x_b, y_b, z_b, phi, rho ;
//   phi = 0;
//
//   for ( int nBlock = 0; nBlock<nBlocks; nBlock++ ){
//     rho = rhoReduced[ nBlock ];
//     x_b = blockX[nBlock];
//     y_b = blockY[nBlock];
//     z_b = blockZ[nBlock];
//     phi -= rsqrt( x_b*x_b + (y_b-y_wall)*(y_b-y_wall) + (z_b-z_wall)*(z_b-z_wall) ) * rho;
//   }
//   phiWall[ tid ] = phi;
//
// }
//
//
//
// // __global__ void getBounderyPotential( const int nWidth, const int nHeight, const int nDepth,
// // 			      float dx, float dy, float dz, float xMin, float yMin, float zMin,
// // 			      cudaP* rhoAll, float* phiWall ){
// // // 			      float* phiWall_l, float* phiWall_r, float* phiWall_d, float* phiWall_u, float* phiWall_b, float* phiWall_t){
// //   int t_j = blockIdx.x*blockDim.x + threadIdx.x;
// //   int t_i = blockIdx.y*blockDim.y + threadIdx.y;
// //   int t_k = blockIdx.z*blockDim.z + threadIdx.z;
// //   int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
// //   int tid_b = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
// //
// //   const int ny = 8*8*2;
// //   const int nz = 2;
// //
// //   const int nSwipes = 50;
// //
// //   float rho = float ( rhoAll[ tid ] );
// //
// //   float x, y, z, y_wall, z_wall, phi;
// //   x = xMin + t_j*dx;
// //   y = yMin + t_i*dy;
// //   z = zMin + t_k*dz;
// //
// //   int idx_1, idx_2, i, j, swipeCounter, swipeIdx;
// //   swipeIdx = tid / 256;
// //
// //   //Allocate shared memory
// //   __shared__ float wallStripe[ 256 ];
// //
// //   for ( swipeCounter=0; swipeCounter<nSwipes; swipeCounter++ ){
// //     //Initialize shared memory to zero
// //     wallStripe[ tid_b ] = 0;
// //     __syncthreads();
// //
// //     //Initialize the indexes over the tile
// //     idx_1 = tid_b % ny;
// //     idx_2 = tid_b / ny;
// //
// //     //Fill the tile of the wall
// //     for ( j=0; j<nz; j++ ){
// //       z_wall = idx_2*dz + zMin;
// //       for ( i=0; i<ny; i++ ){
// // 	y_wall = idx_1*dy + yMin;
// // 	phi = rsqrt( x*x + (y-y_wall)*(y_wall) + (z-z_wall)*(z-z_wall) ) * rho;
// // 	wallStripe[ idx_2*ny + idx_1 ] += phi;
// // 	idx_1 += 1;
// // 	if ( idx_1 >= ny ) idx_1 = 0;
// //       }
// //       idx_2 += 1;
// //       if ( idx_2 >= nz ) idx_2 = 0;
// //     }
// //
// //     //Write the tile values to global memory
// //     idx_1 = tid_b % ny;
// //     idx_2 = tid_b / ny;
// //     atomicAdd( &phiWall[ swipeIdx*256 + idx_2*ny + idx_1  ], wallStripe[ tid_b ] ) ;
// // //     swipeIdx += 1;
// // //     if ( swipeIdx >= 128 ) swipeIdx = 0;
// //   }
// //
// // }
