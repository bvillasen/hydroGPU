// #include <pycuda-helpers.hpp>
__global__ void copyDensity( double *cnsv, double *rho ){
	int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

	rho[tid] = cnsv[tid];
}

__global__ void FFT_divideK2( const double pi4,
							double *kxfft, double *kyfft, double *kzfft,
				      double *data_re, double *data_im){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

  double kx = kxfft[t_j];
  double ky = kyfft[t_i];
  double kz = kzfft[t_k];
  double k2 = kx*kx + ky*ky + kz*kz;
	if ( abs( k2 ) < 1e-5 ) k2 = 1 ;
  data_re[tid] = -data_re[tid]*pi4/k2;
	data_im[tid] = -data_im[tid]*pi4/k2;
}

__global__ void iterPoissonStep(  const int parity,
				 const int nX, const int nY, const int nZ,
         double dx, double dy, double dz,
         const double omega, const double pi4, const double convEpsilon,
				 double* rho_all, double* phi_all, int* converged ){
  int t_j = 2*(blockIdx.x*blockDim.x + threadIdx.x);
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  //Make a checkboard 3D grid
  if ( t_i%2 == 0 ){
    if ( t_k%2 == parity ) t_j +=1;
  }
  else if ( (t_k+1)%2 == parity ) t_j +=1;
  int tid = t_j + t_i*nX + t_k*nX*blockDim.y*gridDim.y;

	//Set neighbors ids
	int l_indx, r_indx, d_indx, u_indx, b_indx, t_indx;
	l_indx = t_j==0    ?    1 : t_j-1;  //Left
	r_indx = t_j==nX-1 ? nX-2 : t_j+1;  //Right
	d_indx = t_i==0    ?    1 : t_i-1;  //Down
	u_indx = t_i==nY-1 ? nY-2 : t_i+1;  //Up
	b_indx = t_k==0    ?    1 : t_k-1;  //bottom
	t_indx = t_k==nZ-1 ? nZ-2 : t_k+1;  //top

  double rho, phi_c, phi_l, phi_r, phi_d, phi_u, phi_b, phi_t, phi_new;
	rho = rho_all[ tid ];
	phi_c = phi_all[tid];
	phi_l = phi_all[ l_indx + t_i*nX + t_k*nX*blockDim.y*gridDim.y ];
	phi_r = phi_all[ r_indx + t_i*nX + t_k*nX*blockDim.y*gridDim.y ];
	phi_d = phi_all[ t_j + d_indx*nX + t_k*nX*blockDim.y*gridDim.y ];
	phi_u = phi_all[ t_j + u_indx*nX + t_k*nX*blockDim.y*gridDim.y ];
	phi_b = phi_all[ t_j + t_i*nX + b_indx*nX*blockDim.y*gridDim.y ];
	phi_t = phi_all[ t_j + t_i*nX + t_indx*nX*blockDim.y*gridDim.y ];

  phi_new = (1-omega)*phi_c + omega/6*( phi_l + phi_r + phi_d + phi_u + phi_b + phi_t - dx*dx*pi4*rho );
  phi_all[ tid ] = phi_new;


  if ( ( abs( ( phi_new - phi_c ) / phi_c ) > convEpsilon ) ) converged[0] = 0;
	// if ( ( abs( phi_new - phi_c ) > 0.0001 ) ) converged[0] = 0;
}

__global__ void getGravityForce(const int nCells,  const int nW, const int nH, const int nD,
				 double dx, double dy, double dz,
				 double* gForce_x, double* gForce_y, double* gForce_z,
				 double* cnsv, double *phi_all, double *gravWork ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

	//Set neighbors ids
	int l_indx, r_indx, d_indx, u_indx, b_indx, t_indx;
	l_indx = t_j==0    ?    1 : t_j-1;  //Left
	r_indx = t_j==nW-1 ? nW-2 : t_j+1;  //Right
	d_indx = t_i==0    ?    1 : t_i-1;  //Down
	u_indx = t_i==nH-1 ? nH-2 : t_i+1;  //Up
	b_indx = t_k==0    ?    1 : t_k-1;  //bottom
	t_indx = t_k==nD-1 ? nD-2 : t_k+1;  //top

  double phi_l, phi_r, phi_d, phi_u, phi_b, phi_t;
	phi_l = phi_all[ l_indx + t_i*nW + t_k*nW*nH ];
	phi_r = phi_all[ r_indx + t_i*nW + t_k*nW*nH ];
	phi_d = phi_all[ t_j + d_indx*nW + t_k*nW*nH ];
	phi_u = phi_all[ t_j + u_indx*nW + t_k*nW*nH ];
	phi_b = phi_all[ t_j + t_i*nW + b_indx*nW*nH ];
	phi_t = phi_all[ t_j + t_i*nW + t_indx*nW*nH ];

  // //Boundary conditions
  // if  ( t_j == 0 )        phi_l = phi_r;
  // if  ( t_j == nWidth-1 ) phi_r = phi_l;
  // if  ( t_i == 0 )        phi_d = phi_u;
  // if  ( t_i == nWidth-1 ) phi_u = phi_d;
  // if  ( t_k == 0 )        phi_b = phi_t;
  // if  ( t_k == nWidth-1 ) phi_t = phi_b;

  //Get partial derivatives for force
  double gField_x, gField_y, gField_z;
  gField_x = ( phi_l - phi_r ) * 0.5 / dx;
  gField_y = ( phi_d - phi_u ) * 0.5 / dy;
  gField_z = ( phi_b - phi_t ) * 0.5 / dz;
	double p_x, p_y, p_z, rho;
	rho = cnsv[ 0*nCells + tid ];
	const int factor = 1;
  gForce_x[ tid ] = gField_x * rho * factor;
  gForce_y[ tid ] = gField_y * rho * factor;
  gForce_z[ tid ] = gField_z * rho * factor;
  // gForce_x[ tid ] = gField_x;
  // gForce_y[ tid ] = gField_y;
  // gForce_z[ tid ] = gField_z;

  // //Get momentum for virtual gravitational work
	p_x = cnsv[ 1*nCells + tid ];
	p_y = cnsv[ 2*nCells + tid ];
	p_z = cnsv[ 3*nCells + tid ];
  gravWork[ tid ] = (p_x * gField_x + p_y * gField_y + p_z * gField_z )*factor;

}
