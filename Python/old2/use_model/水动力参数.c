//Hydrodynamics水动力学参数定义
double rho = 1025; 	double g = 9.81;
double W   = 1.0e4;	double B = 1.0e4; double B_Surface = 1.0e4; double B_Underwater = 1.0e4+50;
double L   = 4; 	double m = 1.0e3;

double Ix = 298; 	double Iy = 3.298e3;	double Iz = 3.4e3;
//x方向 水动力系数定义
static double xG = 0.0; 			static double yG = 0.0; 		static double zG =  30e-3;
static double xB = 0.0; 			static double yB = 0.0; 		static double zB = -30e-3;

static double Xu_ = -4.874e-3;		static double Xwq = -88.246e-3; static double Xvr = 19.42e-3;  static double Xqq = 3.43e-3;
static double Xrr = -2.135e-3;		static double Xpr = -3.118e-3;  static double Xuu = -6.854e-3;
static double Xww = 16.44e-3;		static double Xvv = 6.652e-3;
//z方向 水动力系数定义
static double Yv_ = -29.029e-3; 	static double Ypq = 46.866e-3;  static double Yv = -41.858e-3;
static double Yr = 10.435e-3;		static double Ywp = 68.163e-3;	static double Yr_ = -0.396e-3;
static double Yvw = 7.721e-3; 		static double Yqr = -8.753e-3;	static double Yp1p1 = -510.681e-3;
static double Yr1r1 = 9.189e-3;		static double Yv1r1 = -55.52e-3;
static double Yv1v1 = -28.704e-3;
//z方向 水动力系数定义
static double Zw_ = -126.6e-3; 		static double Zw = -290.9e-3; 	static double Zq = -145.5e-3;
static double Zvp = -31.9e-3;

static double Zq_ = -1.4e-3; 		static double Zpr = -0.396e-3;
static double Zrr = 1.667e-3;
static double Zq1q1 = -13.918e-3; 	static double Zw1q1 = -240.220e-3;
static double Z1w1 = -0.541e-3; 	static double Zww = -40.69e-3; 	static double Z1q1ds = -14.862e-3;
// K方向水动力系数（横摇）
static double Kp_ = -0.5e-3; 		static double Kqr = 0.368e-3; 	static double Kwr = 2.044e-3;
static double Kpq = 2.223e-3; 		static double Kv_ = 0.0;		static double Kp = -1.547e-3;
static double Kr = -0.041e-3; 		static double Kvq = -0.2044e-3; static double Kwp = 5.858e-3;
static double Kp1p1 = -12.47e-3;	static double Mq_ = -5.043e-3; 	static double Muw = -7.4e-3;
static double Mpr = 5.2e-3; 		static double Muq = -49.2e-3;
static double Mrr = 1.321e-3;
static double Mw_ = -1.648e-3; 		static double Mvp = -2.3e-3;
static double Mq1q1 = -20.427e-3;
static double M1w1q = -54.422e-3;	static double Mu1w1 = -1.854e-3;
static double Mw1w1 = -13.329e-3;	static double Mww = 2.621e-3; 	static double M1q1ds = -7.807e-3;
// N方向水动力系数（偏航）
static double Nr_ = -1.050e-3;		static double Nv = -2.872e-3;	static double Nr = -7.335e-3;
static double Nwp = -31.621e-3;		static double Nv_ = -0.396e-3;	static double Np = -8.908e-3;
static double Npq = -16.311e-3;		static double Nqr = 0.417e-3;	static double Nvw = 4.148e-3;
static double Np1p1 = 13.253e-3; 	static double Nr1r1 = -6.453e-3;
static double N1v1r = -13.998e-3;	static double Nv1v1 = 17.440e-3;