#include <cstdlib>
#include <iostream>
// #include "xShare.h"
// #include "xTools.h"
// #include "errors.h"
// #include "utm.h"
#include <math.h>
#include "UUV_Model.h"
// #include "ControllerCommon.h"
// #include "Load.h"
#include <ctime>
#include <pthread.h>
// #include "../include/ArduinoJson.h"

#include <sys/stat.h>
#include <sys/unistd.h>


#include"UUVHull.hpp"
#include"wave_simulation_gpu.hpp"


using namespace std;
// ??????
// CONFIGURATION 	motion_cfg;
OCEANCONFIGURATION 	ocean_current;
// DriversValue  	driversValueCom;
SensorMSG		msg;
Ocean_V_State ocean_v_state;
struct_UuvInf_t uuvinf;//fuhui ??
struct_UuvInf_t uuvinfRev;
structBALANCE_DEVICE_STATUS_info_QUERY_RSP_t VloumeRecv;
FILE* fp = NULL;// ???????????

//?????? UDPRECVPORT ???????????? "10111"
#define UDPRECVPORT    "10111"// UDP????
//#define UDPRECVPORT    "8111"// UDP????
#define UDPSENDPORT    "10112"// UDP????
#define UDPSOCEANENDPORT    "10113"// UDP????
#define UDPRECVSTARTPORT    "3012"
#define UDPRECVVOLUMEPORT    "3013"

const double D2R = PI / 180.0;


int  	gUDPRecvSock  = -1;// UDP?????
int  	gUDPSendSock  = -1;// UDP?????
int     gUDPOceanSendSock  = -1;// ????????UUV??+?????
int  	gUDPRecvSTARTSock  = -1;
int  	gUDPRecvVolumeSock  = -1;

char	Running = 'Y';// ?????'Y'?????'N'????
char	Go = 'G';// ?????'G'?????'N'????
char 	ReloadCfg = 'N';// ?????????'Y'????????

double 	time_counter = 0;// ????????????
long int	 counter = 0;// ????????????

time_t 		now;// ????
//struct tm ? C ? C++ ?????????????????????
//tmnow ????? struct tm ???????????????? struct tm ??????
struct tm 	*tmnow;// ?????????????
char 		filename_Text[100]= {0};

//Hydrodynamics????????
double rho = 1025; 	double g = 9.81;
double W   = 8500;	double B = 1.0e4; double B_Surface = 1.0e4; double B_Underwater = 1.0e4;
double L   = 4; 	double m = W/g;

double Ix = 298; 	double Iy = 3.298e3;	double Iz = 3.4e3;
double Ixy=0.0;double Iyz=0.0;double Ixz=0.0;
//x方向 水动力系数定义
static double xG = 0; 			static double yG = 0.0; 		static double zG =  30e-3;
static double xB = 0; 			static double yB = 0.0; 		static double zB = -30e-3;

// static double Xu_ = -4.874e-3;		static double Xwq = -88.246e-3; static double Xvr = 19.42e-3;  static double Xqq = 3.43e-3;
// static double Xrr = -2.135e-3;		static double Xpr = -3.118e-3;  static double Xuu = -6.854e-3;
// static double Xww = 16.44e-3;		static double Xvv = 6.652e-3;
// //z方向 水动力系数定义
// static double Yv_ = -29.029e-3; 	static double Ypq = 46.866e-3;  static double Yv = -41.858e-3;
// static double Yr = 10.435e-3;		static double Ywp = 68.163e-3;	static double Yr_ = -0.396e-3;
// static double Yvw = 7.721e-3; 		static double Yqr = -8.753e-3;	static double Yp1p1 = -510.681e-3;
// static double Yr1r1 = 9.189e-3;		static double Yv1r1 = -55.52e-3;
// static double Yv1v1 = -28.704e-3;
// //z方向 水动力系数定义
// static double Zw_ = -126.6e-3; 		static double Zw = -290.9e-3; 	static double Zq = -145.5e-3;
// static double Zvp = -31.9e-3;

// static double Zw1w1=-205.7e-3;

// static double Zq_ = -1.4e-3; 		static double Zpr = -0.396e-3;
// static double Zrr = 1.667e-3;
// static double Zq1q1 = -13.918e-3; 	static double Zw1q1 = -240.220e-3;
// static double Z1w1 = -0.541e-3; 	static double Zww = -40.69e-3; 	static double Z1q1ds = -14.862e-3;
// // K方向水动力系数（横摇）
// static double Kp_ = -0.5e-3; 		static double Kqr = 0.368e-3; 	static double Kwr = 2.044e-3;
// static double Kpq = 2.223e-3; 		static double Kv_ = 0.0;		static double Kp = -1.547e-3;
// static double Kr = -0.041e-3; 		static double Kvq = -0.2044e-3; static double Kwp = 5.858e-3;
// static double Kp1p1 = -12.47e-3;	static double Mq_ = -5.043e-3; 	static double Muw = -7.4e-3;
// static double Mpr = 5.2e-3; 		static double Muq = -49.2e-3;
// static double Mrr = 1.321e-3;
// static double Mw_ = -1.648e-3; 		static double Mvp = -2.3e-3;
// static double Mq1q1 = -20.427e-3;
// static double M1w1q = -54.422e-3;	static double Mu1w1 = -1.854e-3;
// static double Mw1w1 = -13.329e-3;	static double Mww = 2.621e-3; 	static double M1q1ds = -7.807e-3;
// // N方向水动力系数（偏航）
// static double Nr_ = -1.050e-3;		static double Nv = -2.872e-3;	static double Nr = -7.335e-3;
// static double Nwp = -31.621e-3;		static double Nv_ = -0.396e-3;	static double Np = -8.908e-3;
// static double Npq = -16.311e-3;		static double Nqr = 0.417e-3;	static double Nvw = 4.148e-3;
// static double Np1p1 = 13.253e-3; 	static double Nr1r1 = -6.453e-3;
// static double N1v1r = -13.998e-3;	static double Nv1v1 = 17.440e-3;
// // ????
// static double Xdsds = -3.370e-3;	static double Xdrdr = -7.578e-3;
// static double Ydr = 14.409e-3;		static double Zds = -14.114e-3;
// static double Mds = -9.232e-3;		static double Ndr = 2 * 9.777e-3;


static double dcoef=1;
static double Xu_=-4.874e-3;static double Xwq=-88.246e-3;static double Xvr=19.42e-3;static double Xqq=3.43e-3;
static double Xrr=-2.135e-3;static double Xpp=0;static double Xpr=-3.118e-3;static double Xuu=-6.854e-3;
static double Xww= 16.44e-3;static double Xvv=6.652e-3;	
static double Xvp=0;

static double Yv_=-29.029e-3;static double Ypq=46.866e-3;static double Yv=-41.858e-3*dcoef;static double Yr=10.435e-3;
static double Ywp=68.163e-3;static double Yvv=-28.704e-3*dcoef;static double Yvr=-55.520e-3;
static double Yr_=-0.396e-3;static double Yp=-16.667e-3;static double Yvq=0;static double Yp_=0;          
static double Ywr=0;		

static double Yvw=7.721e-3;static double Yqr=-8.753e-3;
static double Yp1p1=-510.681e-3;static double Yr1r1=9.189e-3;static double Yv1r1=-55.52e-3;static double Yuu=0;
static double Yv1v1=-28.704e-3;static double Y1r1dr=0;

static double Zw_=-126.6e-3;static double Zw=-290.9e-3;static double Zq=-145.5e-3;static double Zvp=-31.9e-3;
static double Zw1w1=-205.7e-3;static double Zuu=0;
static double Zq_=-1.4e-3;static double Zpp=0;			
static double Zpr=-0.396e-3;	
static double Zrr=1.667e-3;
static double Zvr=0;static double Zvv=0;      
static double Zq1q1=-13.918e-3;static double Zw1q1=-240.220e-3;	
static double Zww=-40.69e-3;static double Z1q1ds=-14.862e-3;static double Z1w1=-0.541e-3;
static double Z0_=2e-3;

static double Kp_=-0.5e-3;static double Kqr=0.368e-3;static double Kwr=2.044e-3;static double Kvv=0;       
static double Kpp=-30e-3;static double Kr_=0.0;		
static double Kpq=2.223e-3;static double Kv_=0.0;          
static double Kp=-1.547e-3;static double Kr=-0.041e-3;static double Kvq=-0.2044e-3;static double Kwp=5.858e-3;      
static double Kv=0;static double Kvw=0;
static double Kp1p1=-12.47e-3;static double Kr1r1=0;static double Kuu=0;
static double Kv1v1=0;static double Kdr=0;

static double Mq_=-5.043e-3;static double Muw=-7.4e-3;static double Mpr=5.2e-3;static double Muq=-49.2e-3;
static double Muu=0;			
static double Mqq=-20.427e-3;
static double Mpp=0;			
static double Mrr=1.321e-3;static double Mw_=-1.648e-3;static double Mvp=-2.3e-3;        
static double Mvr=0;static double Mvv=0;
static double Mq1q1=-20.427e-3;static double M1w1q=-54.422e-3;static double Mu1w1=-1.854e-3;
static double Mw1w1=-13.329e-3;	
static double Mww=2.621e-3;static double M1q1ds=-7.807e-3;
static double M0_=5e-3;

static double Nr_=-1.050e-3;static double Nv=-2.872e-3*dcoef;static double Nr=-7.335e-3;static double Nwp=-31.621e-3;   
static double Nrr=-6.453e-3;static double Nvr=-13.998e-3*dcoef;static double Nvv=17.440e-3*dcoef;
static double Nv_=-0.396e-3;static double Np=-8.908e-3;static double Nvq=0;
static double Np_=-8.908e-3;			
static double Npq=-16.311e-3;static double Nqr=0.417e-3;
static double Nwr=0;			
static double Nvw=4.148e-3;
static double Np1p1=13.253e-3;static double Nr1r1=-6.453e-3;static double N1v1r=-13.998e-3;
static double Nuu=0;			
static double Nv1v1=17.440e-3;static double N1r1dr=0;

static double Xdsds=-3.370e-3;	
static double Xdrdr=-7.578e-3;      

static double Ydr=14.409e-3;	      

static double Zds=-14.114e-3;   

static double Mds=-9.232e-3;

static double Ndr=19.777e-3;

static double Bmax=B;


//double Bmax = 5.6732e4;//Bmax = B
// ????????
static double AuxDistance  = 3;// ??????? (m)
static double MainDistance = 0.5;// ?????? (m)

double uc=0, vc=0, wc=0;

//coefficients for X rudder
// ????
double X_delta1_delta1 = -5.0303e-4; double X_delta2_delta2 = -5.0303e-4;
double X_delta3_delta3 = -5.6451e-4; double X_delta4_delta4 = -5.6451e-4;

double Y_delta1 = -3.8375e-4; double Y_delta2 = -3.8375e-4; double Y_delta3 = 4.6028e-4;  double Y_delta4 = 4.6028e-4;
double Z_delta1 = -6.6442e-4; double Z_delta2 = -6.6442e-4; double Z_delta3 = -7.3760e-4; double Z_delta4 = -7.3760e-4;
double K_delta1 = -1.5876e-3; double K_delta2 = -1.5876e-3; double K_delta3 = -2.0030e-3; double K_delta4 = 2.0030e-3;
double M_delta1 = -5.6597e-3; double M_delta2 = -5.6597e-3; double M_delta3 = -6.8421e-3; double M_delta4 = -6.8421e-3;
double N_delta1 =  5.2033e-3; double N_delta2 =  5.2033e-3; double N_delta3 = -6.7854e-3; double N_delta4 = -6.7854e-3;
// ?????????

//model_inputs ????? UUV ?????????????????????????
//model_states ????? UUV ??????????????????????
double model_inputs[8]  = {0, 0, 0, 0, 0, 0, 0, 0};//?????: ????1,????2,?????1,?????2,?1,?2,?3,?4
double model_states[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};//?????: x y z roll pitch yaw u v w p q r lon lat
// ????
double tempM[6][6] = {
		{m-rho/2*pow(L,3)*Xu_, 		0.0, 						0.0, 						0.0, 					m*zG, 						-m*zG},
		{0.0, 						m-rho/2*pow(L,3)*Yv_, 		0.0, 						0.0, 					0.0, 						m*xG-rho/2*pow(L,4)*Yr_},
		{0.0, 						0.0, 						m-rho/2*pow(L,3)*Zw_, 		m* yG, 					-m* xG-rho/2*pow(L,4)*Zq_, 	0.0},
		{0.0, 						-m* zG-rho/2*pow(L,4)*Kv_, 	m* yG, 						Ix-rho/2*pow(L,5)*Kp_, 	0.0,						0.0},
		{m* zG, 					0.0, 						-m* xG-rho/2*pow(L,4)*Mw_, 	0.0, 					Iy-rho/2*pow(L,5)*Mq_,		0.0},
		{-m* yG, 					m* xG-rho/2*pow(L,4)*Nv_, 	0.0, 						0.0,					0.0, 						Iz-rho/2*pow(L,5)*Nr_}
};

double  accAll[12] = {0,0,0,0,0,0,0,0,0,0,0,0};// ?????????????????

void	GetEulerMatrix(double phi_theta_psi[3], double matrix[9] );// ????????????J1?
void	GetTransMatrix(double phi_theta_psi[3], double matrix[9]);// ???????????J2?
double	mainThrusterMinMax(double value);// ???????
double	auxThrusterMinMax(double value);// ????????
double 	rudderMinMax(double value); // ????
double	*commandforce(double command[],double sim_time);// ?????
double	(*Inverse(double Array[6][6]))[6];// ??????
void	matrixMultipy(double a[6][6] , double b[6], double c[6], int ra, int ca, int rb, int cb);// ????
void 	SendingOutStates();// ??????
void 	SendingOutOceanSpeedStates();// ????????
void*	ServerThread(void*);// ????????UDP??
void*	ServerRecvThread(void*);// ????????UDP??
void*	ServerRecvVolumeThread(void*);// ????????UDP??
unsigned short FillBufTagA(unsigned char buf[], unsigned short curBufLen, unsigned char msgID);
unsigned short GetCheckSumUChar( unsigned char buf[], unsigned short iStart, unsigned short iEnd );
OCEANCONFIGURATION LoadOceanCfg(char* filename);
int		sign(double x);

pthread_mutex_t flag_mutex = PTHREAD_MUTEX_INITIALIZER;

string uuv_hull_path{"/home/robot/AAA/UUV_Model/data/UUVHull.vtk"};
// UUVHull uuv_hull(uuv_hull_path);
// WaveForceEstimator wave_force_calculator(uuv_hull.faces);

UUVHull* gp_uuv_hull = nullptr; 
WaveForceEstimator* gp_wave_force_calculator = nullptr;

int main(){

	gp_uuv_hull = new UUVHull(uuv_hull_path);
    
    // 2. 再初始化 WaveCalculator (GPU - cudaMalloc 将在此刻被安全调用)
	// WaveForceEstimator(std::vector<Face>& faces, double Hs=1.5,double T1=5,
    //                     int wave_N=30,int wave_M=20,int s=5,
    //                     double main_theta=0,std::string wave_type="JONSWAP");
	double Hs=0;
	double T1=3;
	int wave_N=30;
	int wave_M=20;
	int s=5;
	double main_theta=60;
	double sea_depth = 20;
	// double Hs=1.5;
	// double T1=5;
	// int wave_N=30;
	// int wave_M=20;
	// int s=5;
	// double main_theta=0;
	// double depth = 100;
	std::string wave_type="JONSWAP";	// JONSWAP   pierson-Moskowitz
    gp_wave_force_calculator = new WaveForceEstimator(gp_uuv_hull->faces,Hs,T1,wave_N,wave_M,s,main_theta,sea_depth,wave_type);
	
	now=time(NULL);// ??????????
	tmnow=localtime(&now);
	char dir_path[256];
    snprintf(dir_path, sizeof(dir_path), "/home/robot/AAA/UUV_Model/data/uuvMotionData");
    if (access(dir_path, F_OK) == -1) 
	{
        mkdir(dir_path, 0777);  // ?????????
    }

	char file_path[256];
    // 1. 构造目录路径
    snprintf(file_path, sizeof(file_path), "/home/robot/AAA/UUV_Model/data/record/motion");
    
    // 2. 使用mkdir -p创建多级目录（无需判断目录是否存在，-p会自动处理）
    // 拼接mkdir -p命令，注意路径用双引号包裹，避免路径含空格等特殊字符出错
    char mkdir_cmd[512];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p \"%s\"", file_path);
    int mkdir_ret = system(mkdir_cmd);
    if (mkdir_ret == -1) {  // 仅判断system函数调用失败，mkdir -p本身即使目录已存在也会返回0
        perror("system mkdir failed");
        return 1;
    }

	sprintf(filename_Text,"/home/robot/AAA/UUV_Model/data/record/motion/Model_State%04d%02d%02d_%02d%02d.csv",tmnow->tm_year+1900,tmnow->tm_mon+1,tmnow->tm_mday,tmnow->tm_hour,tmnow->tm_min);
	unlink(filename_Text);// ?????
	printf("filename_Text: %s\n", filename_Text);
    fp = fopen(filename_Text, "a+");
    if (fp == NULL) 
	{
		perror("open filename_Text");
		return 1;
    }
	fprintf(fp,"t,x,y,z,roll,pitch,heading,u,v,w,p,q,r,lon,lat,left,right,bow_ver,bow_hor,stern_ver,stern_hor,elevator,rudder,\n");// ?????

   double time_counter_max=200.0;

   double (*InverseM)[6];
	while (Running == 'Y')
	 {
		// while(Go == 'N')
		// {
		// 	xSleep(50);
		// 	SendingOutStates();
		// 	SendingOutOceanSpeedStates();
        //     //SendingOut_output_States();
		// }

//		printf("%f\n",GetHeadingLonLat(122,45,122.1,45));
        //??????????T ???????? time_counter
		double *temp_force;
		
		double accB[6]  = { 0, 0, 0, 0, 0, 0 };
		double accW[6]  = { 0, 0, 0, 0, 0, 0 };
		double Euler[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		double Trans[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		double phi_theta_psi[3] = {0, 0, 0 };// ???			//???:????
		double T = 0.05;	//??							//0.05s???????

		time_counter = time_counter + T;

		// model_inputs[0] =100;
		// model_inputs[1] =100;
		// model_inputs[2] = 50;
		// model_inputs[4] = 50;

        // ?????
		temp_force = commandforce(model_inputs,time_counter); //?????8?????: ?????????????????????????????????????????????????; ????UUV?????????????????? x,y,z,p,q,r
        // ????????
		
		InverseM = Inverse(tempM); //?????tempM????

		matrixMultipy(InverseM, temp_force, accB, 6, 6, 6, 1); //??UUV????????????u' v' w' p' q' r'

		
		// ?????
		phi_theta_psi[0] = model_states[3];
		phi_theta_psi[1] = model_states[4];
		phi_theta_psi[2] = model_states[5];

		
		//??????
		GetEulerMatrix(phi_theta_psi, Euler);
		GetTransMatrix(phi_theta_psi, Trans);
        // ????????????
		accW[0] = Euler[0] * model_states[9] + Euler[1] * model_states[10] + Euler[2] * model_states[11];
		accW[1] = Euler[3] * model_states[9] + Euler[4] * model_states[10] + Euler[5] * model_states[11];
		accW[2] = Euler[6] * model_states[9] + Euler[7] * model_states[10] + Euler[8] * model_states[11];

		accW[3] = Trans[0] * model_states[6] + Trans[1] * model_states[7]  + Trans[2] * model_states[8];
		accW[4] = Trans[3] * model_states[6] + Trans[4] * model_states[7]  + Trans[5] * model_states[8];
		accW[5] = Trans[6] * model_states[6] + Trans[7] * model_states[7]  + Trans[8] * model_states[8];
		// ???????
		for (int i = 0; i < 6; i++)
		{
			accAll[i] =  accB[i];	//u' v' w' p' q' r'// ????
		}
		for (int i = 0; i < 6; i++)
		{
			accAll[i+6] =  accW[i];//phi' theta' psi' x' y' z'// ????
		}

		for (int i = 0; i < 6; i++) // u v w p q r
		{
			// ????
			model_states[i + 6] = model_states[i + 6] + T * accAll[i];
		}
		for (int i = 0; i < 3; i++) // x y z
		{
			// ????
			model_states[i] = model_states[i] + T * accAll[i + 6 + 3];
		}
		for (int i = 3; i < 6; i++) // roll pitch yaw
		{
			// ????
			model_states[i] = model_states[i] + T * accAll[i + 3];
		}
        // ?????
		model_states[12] = model_states[12] + T * accAll[10]/(cos(model_states[13]*D2R)*60*1852);  //lon??

		model_states[13] = model_states[13] + T * accAll[9] /111120.0;  //lat??
       // ??????0?2???
		if (model_states[5] < 0)
			model_states[5] = model_states[5] + 2 * PI;
		if (model_states[5] > 2 * PI)
			model_states[5] = model_states[5] - 2 * PI;
       // ?10?????????
		if(counter%10==0)
		{
			printf("\nUUV input: ");
			for(int i = 0;i<8;i++)
			{
				printf("%f ",model_inputs[i]);//model_inputs ????? UUV ?????????????????????????
			}
			printf("\nUUV state: ");
			for(int i = 0;i<14;i++)
			{
				if(i==0) 	printf("x:%f \n",model_states[i]);//model_states ????? UUV ??????????????????????
				if(i==1) 	printf("y:%f \n",model_states[i]);//model_states ????? UUV ??????????????????????
				if(i==2) 	printf("z:%f \n",model_states[i]);//model_states ????? UUV ??????????????????????
				if(i==3) 	printf("roll:%f \n",model_states[i]*57.3);//model_states ????? UUV ??????????????????????
				if(i==4) 	printf("pitch:%f \n",model_states[i]*57.3);//model_states ????? UUV ??????????????????????
				if(i==5) 	printf("yaw:%f \n",model_states[i]*57.3);//model_states ????? UUV ??????????????????????
				if(i==6) 	printf("u:%f \n",model_states[i]);//model_states ????? UUV ??????????????????????
				if(i==7) 	printf("v:%f \n",model_states[i]);//model_states ????? UUV ??????????????????????
				if(i==8) 	printf("w:%f \n",model_states[i]);//model_states ????? UUV ??????????????????????
				if(i==9) 	printf("p:%f \n",model_states[i]);//model_states ????? UUV ??????????????????????
				if(i==10)	printf("q:%f \n",model_states[i]);//model_states ????? UUV ??????????????????????
				if(i==11)	printf("r:%f \n",model_states[i]);//model_states ????? UUV ??????????????????????
				if(i==12)	printf("lon:%f \n",model_states[i]);//model_states ????? UUV ??????????????????????
				if(i==13)	printf("lat:%f \n",model_states[i]);//model_states ????? UUV ??????????????????????
			}

			printf("\nsum simulation time: %fs\n",time_counter);
		}
		// fp=fopen(filename_Text,"a+");//??????a+?????????? filename_Text?????????????????????????????????
		fprintf(fp,"%.8f,",time_counter);//? time_counter ???????????????? 8 ????
		for(int j = 0; j<14; j++){
			fprintf(fp,"%.8f,",model_states[j]);//? model_states ???????????????????? 8 ??????????
		}
		for(int j = 0;j<8;j++){
			fprintf(fp,"%.8f,",model_inputs[j]);//? model_inputs ???????????????????? 8 ???????????
		}
		fprintf(fp,"\n");//????????????????????????
		counter++;
		if(time_counter>=time_counter_max)Running='N';
	}

	delete gp_wave_force_calculator;
    delete gp_uuv_hull;
    printf("simulation end");
	return 0;
}

// //??????
// //?????????
// //????????????J1????????????
void GetEulerMatrix(double phi_theta_psi[3], double matrix[9] )
{
	double ptr[3];
	ptr[0]	  = phi_theta_psi[0];
	ptr[1]	  = phi_theta_psi[1];
	ptr[2] 	  = phi_theta_psi[2];
	matrix[0] = 1;
	matrix[1] = sin(ptr[0]) * tan(ptr[1]);
	matrix[2] = cos(ptr[0]) * tan(ptr[1]);
	matrix[3] = 0;
	matrix[4] = cos(ptr[0]);
	matrix[5] = -sin(ptr[0]);
	matrix[6] = 0;
	matrix[7] = sin(ptr[0]) / cos(ptr[1]);
	matrix[8] = cos(ptr[0]) / cos(ptr[1]);
}
// //?????????
// //????????????J2????????????
void GetTransMatrix(double phi_theta_psi[3], double matrix[9])
{
   double ptr[3];
   ptr[0] 	 = phi_theta_psi[0];
   ptr[1] 	 = phi_theta_psi[1];
   ptr[2] 	 = phi_theta_psi[2];
   matrix[0] = cos(ptr[1]) * cos(ptr[2]);
   matrix[1] = sin(ptr[0]) * sin(ptr[1]) * cos(ptr[2]) - cos(ptr[0]) * sin(ptr[2]);
   matrix[2] = cos(ptr[0]) * sin(ptr[1]) * cos(ptr[2]) + sin(ptr[0]) * sin(ptr[2]);
   matrix[3] = cos(ptr[1]) * sin(ptr[2]);
   matrix[4] = sin(ptr[0]) * sin(ptr[1]) * sin(ptr[2]) + cos(ptr[0]) * cos(ptr[2]);
   matrix[5] = cos(ptr[0]) * sin(ptr[1]) * sin(ptr[2]) - sin(ptr[0]) * cos(ptr[2]);
   matrix[6] = -sin(ptr[1]);
   matrix[7] = sin(ptr[0]) * cos(ptr[1]);
   matrix[8] = cos(ptr[0]) * cos(ptr[1]);
}

int sign(double x) {
	int a;
	if (x > 0) {
		a = 1;
	} else if (x == 0) {
		a = 0;
	} else {
		a = -1;
	}
	return a;
}



// 新增函数：计算从 NED (世界) 到 Body 的旋转矩阵 R_nb 的转置 (即 R_bn)
// 用于把世界系下的力(如浮力)转到船体坐标系
void GetRotationMatrixTransposed(double phi, double theta, double psi, double R[9]) {
    double cphi = cos(phi), sphi = sin(phi);
    double cth  = cos(theta), sth  = sin(theta);
    double cpsi = cos(psi), spsi = sin(psi);

    // R_body_to_world (R_nb) 的转置 = R_world_to_body
    // Row 1
    R[0] = cpsi * cth;
    R[1] = spsi * cth;
    R[2] = -sth;
    // Row 2
    R[3] = cpsi * sth * sphi - spsi * cphi;
    R[4] = spsi * sth * sphi + cpsi * cphi;
    R[5] = cth * sphi;
    // Row 3
    R[6] = cpsi * sth * cphi + spsi * sphi;
    R[7] = spsi * sth * cphi - cpsi * sphi;
    R[8] = cth * cphi;
}



// //??????????????????????? UUV ???????? UUV ???????????????????????????????????????????
// //?????????? temp_force ????????????
double* commandforce(double command[8],double now_time)//double command[8]????? 8 ???????????????????????????UUV???????????????????????????????????????
{

	static double temp_force[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	double u, v, w;
	double p, q, r;
	double phi, theta, psi;
	double depth;

	//?????????
	double V_north=0;  
    double V_east=0;   
    double V_down = 0;  
	double phi_theta_psi1[3] = {0, 0, 0 };
	double Trans1[9]={0,0,0,0,0,0,0,0,0};
	
	phi_theta_psi1[0] = model_states[3];
	phi_theta_psi1[1] = model_states[4];
	phi_theta_psi1[2] = model_states[5];
 
	u = model_states[6]; v = model_states[7]; w = model_states[8];
	p = model_states[9]; q = model_states[10]; r = model_states[11];

	depth = model_states[2]; phi = model_states[3]; theta = model_states[4]; psi = model_states[5];

	double StaticForceMoment[6]  = { 0, 0, 0, 0, 0, 0 };  //????????????
	double OtherHydrodynamic[6]  = { 0, 0, 0, 0, 0, 0 };  //??????????????
	double ControlPaneForce[6]   = { 0, 0, 0, 0, 0, 0 };  //????????????????????????
	double EnvironForceMoment[6] = { 0, 0, 0, 0, 0, 0 }; //??????????

	std::vector<double> wave_result(7);
	// wave_result=wave_force_calculator.compute_wave_force(sim_time,
	// 													model_states[0],model_states[1],model_states[2],
	// 													model_states[3],model_states[4],model_states[5]);
	if (gp_wave_force_calculator) {
		wave_result = gp_wave_force_calculator->compute_wave_force(now_time,
														model_states[0], model_states[1], model_states[2],
														model_states[3], model_states[4], model_states[5]);
	} else {
		// 处理未初始化的情况，或者直接报错
		printf("Error: Wave calculator not initialized!\n");
		return NULL; // 或者其他错误处理
	}
	double wet_ratio = wave_result[6]; 
	wet_ratio = max(wet_ratio, 0.2);

//  Limitations of control inputs without consider about inertial of thruster and rudder
//??????????????????????????????
    V_north=ocean_current.Ocean_UC;
	V_east=ocean_current.Ocean_VC;
	//printf("?????????UC:%f ",V_north);
	//printf("?????????VC:%f ",V_east);

	// command[0] = mainThrusterMinMax(command[0]);
	// command[1] = mainThrusterMinMax(command[1]);

	// command[2] = auxThrusterMinMax(command[2]);
	// command[3] = auxThrusterMinMax(command[3]);
	// command[4] = auxThrusterMinMax(command[4]);
	// command[5] = auxThrusterMinMax(command[5]);

	// command[6] = rudderMinMax(command[6]);
	// command[7] = rudderMinMax(command[7]);

//???????? B ???
	// if(depth>0)
	// 	B = B_Underwater;
	// else if(depth<=0)
	// B = B_Underwater+depth*10000;
		//??????
	GetTransMatrix(phi_theta_psi1, Trans1);
		
// ??????????????????
	double R_earth_to_body[9] = {
		Trans1[0], Trans1[3], Trans1[6],
		Trans1[1], Trans1[4], Trans1[7],
		Trans1[2], Trans1[5], Trans1[8]};
	// GetEulerMatrix?????????????????????????????????????????????
	// ?????GetTransMatrix??????GetEulerMatrix?

// ???????????
	uc = R_earth_to_body[0] * V_north + R_earth_to_body[1] * V_east + R_earth_to_body[2] * V_down;
	vc = R_earth_to_body[3] * V_north + R_earth_to_body[4] * V_east + R_earth_to_body[5] * V_down;
	wc = R_earth_to_body[6] * V_north + R_earth_to_body[7] * V_east + R_earth_to_body[8] * V_down;

	// 如果在空气中 (wet_ratio=0)，洋流对船体没有作用，相对速度基准应变为 0
	double uc_eff = uc * wet_ratio;
	double vc_eff = vc * wet_ratio;
	double wc_eff = wc * wet_ratio;
	// 水动力(阻尼/附加质量力)与密度成正比。空气中 rho 近似为 0 (相对于水的 1025)
	double eff_rho = rho * wet_ratio; 

	tempM[0][0]=m-eff_rho/2*pow(L,3)*Xu_;
	tempM[1][1]=m-eff_rho/2*pow(L,3)*Yv_;
	tempM[1][5]=m*xG-eff_rho/2*pow(L,4)*Yr_;
	tempM[2][2]=m-eff_rho/2*pow(L,3)*Zw_;
	tempM[2][4]=-m* xG-eff_rho/2*pow(L,4)*Zq_;
	tempM[3][1]=-m* zG-eff_rho/2*pow(L,4)*Kv_;
	tempM[3][3]=Ix-eff_rho/2*pow(L,5)*Kp_;
	tempM[4][2]=-m* xG-eff_rho/2*pow(L,4)*Mw_;
	tempM[4][4]=Iy-eff_rho/2*pow(L,5)*Mq_;
	tempM[5][1]=m* xG-eff_rho/2*pow(L,4)*Nv_;
	tempM[5][5]=Iz-eff_rho/2*pow(L,5)*Nr_;


//?????????????????
	// cout<<"uc:"<<uc<<endl;
	// cout<<"vc:"<<vc<<endl;
	// cout<<"wc:"<<wc<<endl;

//????????????????????

	// 提取 World Frame 下的力和力矩
    double F_world[3] = { wave_result[0], wave_result[1], wave_result[2] };
    double M_world[3] = { wave_result[3], wave_result[4], wave_result[5] };
	// EnvironForceMoment[0] = R_earth_to_body[0]*F_world[0] + R_earth_to_body[1]*F_world[1] + R_earth_to_body[2]*F_world[2];
    // EnvironForceMoment[1] = R_earth_to_body[3]*F_world[0] + R_earth_to_body[4]*F_world[1] + R_earth_to_body[5]*F_world[2];
    // EnvironForceMoment[2] = R_earth_to_body[6]*F_world[0] + R_earth_to_body[7]*F_world[1] + R_earth_to_body[8]*F_world[2];

    // // 转换力矩 Moment_body = R_e2b * Moment_world
    // EnvironForceMoment[3] = R_earth_to_body[0]*M_world[0] + R_earth_to_body[1]*M_world[1] + R_earth_to_body[2]*M_world[2];
    // EnvironForceMoment[4] = R_earth_to_body[3]*M_world[0] + R_earth_to_body[4]*M_world[1] + R_earth_to_body[5]*M_world[2];
    // EnvironForceMoment[5] = R_earth_to_body[6]*M_world[0] + R_earth_to_body[7]*M_world[1] + R_earth_to_body[8]*M_world[2];

	double R_world_to_body[9];
	GetRotationMatrixTransposed(phi, theta, psi, R_world_to_body);

	EnvironForceMoment[0] = R_world_to_body[0]*F_world[0] + R_world_to_body[1]*F_world[1] + R_world_to_body[2]*F_world[2];
	EnvironForceMoment[1] = R_world_to_body[3]*F_world[0] + R_world_to_body[4]*F_world[1] + R_world_to_body[5]*F_world[2];
	EnvironForceMoment[2] = R_world_to_body[6]*F_world[0] + R_world_to_body[7]*F_world[1] + R_world_to_body[8]*F_world[2];

	EnvironForceMoment[3] = R_world_to_body[0]*M_world[0] + R_world_to_body[1]*M_world[1] + R_world_to_body[2]*M_world[2];
	EnvironForceMoment[4] = R_world_to_body[3]*M_world[0] + R_world_to_body[4]*M_world[1] + R_world_to_body[5]*M_world[2];
	EnvironForceMoment[5] = R_world_to_body[6]*M_world[0] + R_world_to_body[7]*M_world[1] + R_world_to_body[8]*M_world[2];

	B=0;
	StaticForceMoment[0] = -(W - B) * sin(theta);
	StaticForceMoment[1] =  (W - B) * cos(theta) * sin(phi);
	StaticForceMoment[2] =  (W - B) * cos(theta) * cos(phi);

	StaticForceMoment[3] =  (yG * W - yB * B) * cos(theta) * cos(phi) - (zG * W - zB * B) * cos(theta) * sin(phi);
	StaticForceMoment[4] = -(xG * W - xB * B) * cos(theta) * cos(phi) - (zG * W - zB * B) * sin(theta);
	StaticForceMoment[5] =  (xG * W - xB * B) * cos(theta) * sin(phi) + (yG * W - yB * B) * sin(theta);
//???????????????????????????????????????????????? UUV ???????????
	OtherHydrodynamic[0] = StaticForceMoment[0]
	   + m * (v - vc_eff) * r - m * (w - wc_eff) * q + m * xG * (q * q + r * r) - m * yG * p * q - m * zG * p * r
	   + eff_rho / 2 * pow(L, 4) * (Xqq * q * q + Xrr * r * r + Xpr * p * r)
	   + eff_rho / 2 * pow(L, 3) * (Xwq * (w - wc_eff) * q + Xvr * (v - vc_eff) * r)
	   + eff_rho / 2 * pow(L, 2) * (Xvv * (v - vc_eff) * (v - vc_eff) + Xww * (w - wc_eff) * (w - wc_eff) + Xuu * (u - uc_eff) * fabs(u - uc_eff));

	OtherHydrodynamic[1] = StaticForceMoment[1]
	   - m * (u - uc_eff) * r + m * (w - wc_eff) * p - m * xG * p * q + m * yG * (p * p + r * r) - m * zG * q * r
	   + eff_rho / 2 * pow(L, 4) * (Yp1p1 * p * fabs(p) + Ypq * p * q + Yqr * q * r + Yr1r1 * r * fabs(r))
	   + eff_rho / 2 * pow(L, 3) * (Yr * (u - uc_eff) * r + Ywp * (w - wc_eff) * p + Yv1r1 * sign(v - vc_eff) * sqrt(fabs((w - wc_eff) * (w - wc_eff) + (v - vc_eff) * (v - vc_eff))) * fabs(r))
	   + eff_rho / 2 * pow(L, 2) * (Yv * (u - uc_eff) * (v - vc_eff) + Yvw * (v - vc_eff) * (w - wc_eff) + Yv1v1 * (v - vc_eff) * sqrt(fabs((w - wc_eff) * (w - wc_eff) + (v - vc_eff) * (v - vc_eff))));

	OtherHydrodynamic[2] = StaticForceMoment[2]
       + m * (u - uc_eff) * q - m * (v - vc_eff) * p - m * xG * p * r - m * yG * q * r + m * zG * (p * p + q * q)
	   + eff_rho / 2 * pow(L, 4) * (Zrr * r * r + Zpr * p * r + Zq1q1 * q * fabs(q))
	   + eff_rho / 2 * pow(L, 3) * (Zq * fabs(u - uc_eff) * q + Zvp * (v - vc_eff) * p + Zw1q1 * sign(w - wc_eff) * sqrt(pow(fabs(v - vc_eff), 2) + pow(fabs(w - wc_eff), 2)) * fabs(q))
	   + eff_rho / 2 * pow(L, 2) * (Zw * fabs(u - uc_eff) * (w - wc_eff) + Z1w1 * fabs(u - uc_eff) * fabs(w - wc_eff) + Zww * fabs(w - wc_eff) * sqrt(fabs((v - vc_eff) * (v - vc_eff) + (w - wc_eff) * (w - wc_eff))));

	OtherHydrodynamic[3] = StaticForceMoment[3]
	   - (Iz - Iy) * q * r - m * yG * (-(u - uc_eff) * q + (v - vc_eff) * p) + m * zG * ((u - uc_eff) * r - (w - wc_eff) * p)
	   + eff_rho / 2 * pow(L, 5) * (Kp1p1 * p * fabs(p) + Kqr * q * r + Kpq * p * q)
	   + eff_rho / 2 * pow(L, 4) * (Kp * (u - uc_eff) * p + Kr * (u - uc_eff) * r + Kvq * (v - vc_eff) * q + Kwp * (w - wc_eff) * p + Kwr * (w - wc_eff) * r);

	OtherHydrodynamic[4] = StaticForceMoment[4]
	   - (Ix - Iz) * p * r + m * xG * (-(u - uc_eff) * q + (v - vc_eff) * p) - m * zG * (-(v - vc_eff) * r + (w - wc_eff) * q)
	   + eff_rho / 2 * pow(L, 5) * (Mq1q1 * q * fabs(q) + Mrr * r * r + Mpr * p * r)
	   + eff_rho / 2 * pow(L, 4) * (Muq * fabs(u - uc_eff) * q + Mvp * (v - vc_eff) * p + M1w1q * sqrt(fabs((v - vc_eff) * (v - vc_eff) + (w - wc_eff) * (w - wc_eff))) * q)
	   + eff_rho / 2 * pow(L, 3) * (Muw * fabs(u - uc_eff) * (w - wc_eff) + Mu1w1 * fabs(u - uc_eff) * fabs(w - wc_eff) + Mw1w1 * (w - wc_eff) * sqrt(fabs((v - vc_eff) * (v - vc_eff) + (w - wc_eff) * (w - wc_eff)))
	   + Mww * fabs(w - wc_eff) * sqrt(fabs((v - vc_eff) * (v - vc_eff) + (w - wc_eff) * (w - wc_eff))));
	// OtherHydrodynamic[4] = OtherHydrodynamic[4] - 100*u*u;  // TODO: Shitty Method

	OtherHydrodynamic[5] = StaticForceMoment[5]
	   - (Iy - Ix) * p * q - m * xG * ((u - uc_eff) * r - (w - wc_eff) * p) + m * yG * (-(v - vc_eff) * r + (w - wc_eff) * q)
	   + eff_rho / 2 * pow(L, 5) * (Np1p1 * p * fabs(p) + Nr1r1 * r * fabs(r) + Npq * p * q + Nqr * q * r)
	   + eff_rho / 2 * pow(L, 4) * (Np * (u - uc_eff) * p + Nr * (u - uc_eff) * (r) + Nwp * (w - wc_eff) * p + N1v1r * sqrt(fabs((v - vc_eff) * (v - vc_eff) + (w - wc_eff) * (w - wc_eff))) * r)
	   + eff_rho / 2 * pow(L, 3) * (Nv * (u - uc_eff) * (v - vc_eff) + Nvw * (v - vc_eff) * (w - wc_eff) + Nv1v1 * (v - vc_eff) * sqrt(fabs((v - vc_eff) * (v - vc_eff) + (w - wc_eff) * (w - wc_eff))));




	// OtherHydrodynamic[0]=StaticForceMoment[0]+
	//    m*(v-vc_eff)*r-m*(w-wc_eff)*q-m*xG*(q*q+r*r)-m*yG*p*q-m*zG*p*r+
	//    eff_rho/2*pow(L,4)*(Xqq*q*q+Xrr*r*r+Xpr*p*r)+
	//    eff_rho/2*pow(L,3)*(Xwq*(w-wc_eff)*q+Xvr*(v-vc_eff)*r)+
	//    eff_rho/2*pow(L,2)*(Xvv*(v-vc_eff)*(v-vc_eff)+Xww*(w-wc_eff)*(w-wc_eff)+Xuu*(u-uc_eff)*abs(u-uc_eff));
	// OtherHydrodynamic[1]=StaticForceMoment[1]-
	//    m*(u-uc_eff)*r+m*(w-wc_eff)*p-m*xG*p*q+m*yG*(p*p+r*r)-m*zG*q*r+
	//    eff_rho/2*pow(L,4)*(Yp1p1*p*abs(p)+Ypq*p*q+Yqr*q*r+Yr1r1*r*abs(r))+
	//    eff_rho/2*pow(L,3)*(Yr*(u-uc_eff)*r+Yvq*(v-vc_eff)*q+Ywp*(w-wc_eff)*p+Ywr*(w-wc_eff)*r+
	// 	   Yv1r1*sign(v-vc_eff)*sqrt(abs((w-wc_eff)*(w-wc_eff)+(v-vc_eff)*(v-vc_eff)))*abs(r))+
	//    eff_rho/2*pow(L,2)*(Yuu*(u-uc_eff)*(u-uc_eff)+Yv*(u-uc_eff)*(v-vc_eff)+Yvw*(v-vc_eff)*(w-wc_eff)+
	// 	   Yv1v1*(v-vc_eff)*sqrt(abs((w-wc_eff)*(w-wc_eff)+(v-vc_eff)*(v-vc_eff))));
	// OtherHydrodynamic[2]=StaticForceMoment[2]+
	//    m*(u-uc_eff)*q-m*(v-vc_eff)*p-m*xG*p*r-m*yG*q*r+m*zG*(p*p+q*q)+
 	//    eff_rho/2*pow(L,4)*(Zpp*p*p+Zrr*r*r+Zpr*p*r+Zq1q1*q*abs(q))+
 	//    eff_rho/2*pow(L,3)*(Zq*abs(u-uc_eff)*q+Zvr*(v-vc_eff)*r+Zvp*(v-vc_eff)*p+
 	// 	   Zw1q1*sign(w-wc_eff)*sqrt((v-vc_eff)*(v-vc_eff)+(w-wc_eff)*(w-wc_eff))*abs(q))+
	//    eff_rho/2*pow(L,2)*(Zuu*(u-uc_eff)*(u-uc_eff)+Zw*abs(u-uc_eff)*(w-wc_eff)+Z1w1*abs(u-uc_eff)*abs(w-wc_eff)+
	// 	   Zvv*(v-vc_eff)*(v-vc_eff)+Zww*abs(w-wc_eff)*sqrt(abs((v-vc_eff)*(v-vc_eff)+(w-wc_eff)*(w-wc_eff)))+
	// 	   Zw1w1*(w-wc_eff)*sqrt(abs((v-vc_eff)*(v-vc_eff)+(w-wc_eff)*(w-wc_eff))));
	// OtherHydrodynamic[3]=StaticForceMoment[3]-
	//    (Iz-Iy)*q*r-Ixy*p*r+Iyz*(q*q-r*r)+Ixz*p*q-m*yG*(-(u-uc_eff)*q+(v-vc_eff)*p)+m*zG*((u-uc_eff)*r-(w-wc_eff)*p)+
    //    eff_rho/2*pow(L,5)*(Kp1p1*p*abs(p)+Kqr*q*r+Kpq*p*q+Kr1r1*r*abs(r))+
	//    eff_rho/2*pow(L,4)*(Kp*(u-uc_eff)*p+Kr*(u-uc_eff)*r+Kvq*(v-vc_eff)*q+Kwp*(w-wc_eff)*p+Kwr*(w-wc_eff)*r)+
	//    eff_rho/2*pow(L,3)*(Kuu*(u-uc_eff)*(u-uc_eff)+Kv*(u-uc_eff)*(v-vc_eff)+
	// 	   Kv1v1*(v-vc_eff)*sqrt(abs((v-vc_eff)*(v-vc_eff)+(w-wc_eff)*(w-wc_eff)))+
	// 	   Kvw*(v-vc_eff)*(w-wc_eff));
	// OtherHydrodynamic[4]=StaticForceMoment[4]-
	//    (Ix-Iz)*p*r+Ixy*q*r-Iyz*p*q-Ixz*(p*p-r*r)+m*xG*(-(u-uc_eff)*q+(v-vc_eff)*p)-m*zG*(-(v-vc_eff)*r+(w-wc_eff)*q)+
	//    eff_rho/2*pow(L,5)*(Mpp*p*p+Mq1q1*q*abs(q)+Mrr*r*r+Mpr*p*r)+
	//    eff_rho/2*pow(L,4)*(Muq*abs(u-uc_eff)*q+Mvr*(v-vc_eff)*r+Mvp*(v-vc_eff)*p+
	// 	   M1w1q*sqrt(abs((v-vc_eff)*(v-vc_eff)+(w-wc_eff)*(w-wc_eff)))*q)+
	//    eff_rho/2*pow(L,3)*(Muu*(u-uc_eff)*(u-uc_eff)+Muw*abs(u-uc_eff)*(w-wc_eff)+Mu1w1*abs(u-uc_eff)*abs(w-wc_eff)+
	// 	   Mw1w1*(w-wc_eff)*sqrt(abs((v-vc_eff)*(v-vc_eff)+(w-wc_eff)*(w-wc_eff)))+Mvv*(v-vc_eff)*(v-vc_eff)+
	// 	   Mww*abs(w-wc_eff)*sqrt(abs((v-vc_eff)*(v-vc_eff)+(w-wc_eff)*(w-wc_eff)))); 
	// OtherHydrodynamic[5]=StaticForceMoment[5]-
	//    (Iy-Ix)*p*q+Ixy*(p*p-q*q)+Iyz*p*r-Ixz*q*r-m*xG*((u-uc_eff)*r-(w-wc_eff)*p)+m*yG*(-(v-vc_eff)*r+(w-wc_eff)*q)+
 	//    eff_rho/2*pow(L,5)*(Np1p1*p*abs(p)+Nr1r1*r*abs(r)+Npq*p*q+Nqr*q*r)+
	//    eff_rho/2*pow(L,4)*(Np*(u-uc_eff)*p+Nr*(u-uc_eff)*(r)+Nwr*(w-wc_eff)*r+Nwp*(w-wc_eff)*p+Nvq*(v-vc_eff)*q+
	// 	   N1v1r*sqrt(abs((v-vc_eff)*(v-vc_eff)+(w-wc_eff)*(w-wc_eff)))*r)+
	//    eff_rho/2*pow(L,3)*(Nuu*(u-uc_eff)*(u-uc_eff)+Nv*(u-uc_eff)*(v-vc_eff)+Nvw*(v-vc_eff)*(w-wc_eff)+
	// 	   Nv1v1*(v-vc_eff)*sqrt(abs((v-vc_eff)*(v-vc_eff)+(w-wc_eff)*(w-wc_eff))));

	//???????????????????????????????????????????
	ControlPaneForce[0] =
		command[0] + command[1] + eff_rho / 2 * pow(L, 2) * fabs(u - uc_eff) * (u - uc_eff) * (Xdrdr * command[7] * command[7] + Xdsds * command[6] * command[6]);
//            + eff_rho / 2 * 1 * (X_delta1_delta1 * (u - uc_eff) * (u - uc_eff) * Math.Abs(x_rudder_inputs[0])
//            + X_delta2_delta2 * (u - uc_eff) * (u - uc_eff) * Math.Abs(x_rudder_inputs[1])
//            + X_delta3_delta3 * (u - uc_eff) * (u - uc_eff) * Math.Abs(x_rudder_inputs[2])
//            + X_delta4_delta4 * (u - uc_eff) * (u - uc_eff) * Math.Abs(x_rudder_inputs[3]));

	ControlPaneForce[1] =
		eff_rho / 2 * pow(L, 2) * Ydr * fabs(u - uc_eff) * (u - uc_eff) * command[7] + command[3] + command[5];
//            + eff_rho / 2 * 1 * (Y_delta1 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[0]
//            + Y_delta2 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[1]
//            + Y_delta3 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[2]
//            + Y_delta4 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[3]);

	ControlPaneForce[2] =
		eff_rho / 2 * pow(L, 2) * Zds * fabs(u - uc_eff) * (u - uc_eff) * command[6] + command[2] + command[4] + eff_rho / 2 * pow(L, 3) * Z1q1ds * (u - uc_eff) * fabs(q) * command[6];
//            + eff_rho / 2 * 1 * (Z_delta1 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[0]
//            + Z_delta2 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[1]
//            + Z_delta3 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[2]
//            + Z_delta4 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[3]);

	ControlPaneForce[3] = 0;
//            + eff_rho / 2 * 1 * (K_delta1 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[0]
//            + K_delta2 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[1]
//            + K_delta3 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[2]
//            + K_delta4 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[3]);

	ControlPaneForce[4] =
		eff_rho / 2 * pow(L, 3) * Mds * fabs(u - uc_eff) * (u - uc_eff) * command[6] -command[2] * AuxDistance / 2 + command[4] * AuxDistance / 2 + eff_rho / 2 * pow(L, 4) * M1q1ds * (u - uc_eff) * fabs(q) * command[6];
//            + eff_rho / 2 * 1 * (M_delta1 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[0]
//            + M_delta2 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[1]
//            + M_delta3 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[2]
//            + M_delta4 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[3]);

	ControlPaneForce[5] =
		eff_rho / 2 * pow(L, 3) * Ndr * fabs(u - uc_eff) * (u - uc_eff) * command[7] + command[0] * MainDistance / 2 - command[1] * MainDistance / 2 + command[3] * AuxDistance / 2 - command[5] * AuxDistance / 2;
//            + eff_rho / 2 * 1 * (N_delta1 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[0]
//            + N_delta2 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[1]
//            + N_delta3 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[2]
//            + N_delta4 * (u - uc_eff) * (u - uc_eff) * x_rudder_inputs[3]);
//????????????????????????????????????? temp_force ????
	temp_force[0] = ControlPaneForce[0] + OtherHydrodynamic[0] + EnvironForceMoment[0];
	temp_force[1] = ControlPaneForce[1] + OtherHydrodynamic[1] + EnvironForceMoment[1];
	temp_force[2] = ControlPaneForce[2] + OtherHydrodynamic[2] + EnvironForceMoment[2]; 
	temp_force[3] = ControlPaneForce[3] + OtherHydrodynamic[3] + EnvironForceMoment[3]; 
	temp_force[4] = ControlPaneForce[4] + OtherHydrodynamic[4] + EnvironForceMoment[4];
	temp_force[5] = ControlPaneForce[5] + OtherHydrodynamic[5] + EnvironForceMoment[5];
//double*???????????????????? temp_force ??????????? 6 ???????????????? command[8] ??? UUV ???????????????????????????????? X?Y?Z ?????? X?Y?Z ??????????????????? UUV ??????
	
	cout<<StaticForceMoment[0]<<" "<<StaticForceMoment[1]<<" "<<StaticForceMoment[2]<<" "<<StaticForceMoment[3]<<" "<<StaticForceMoment[4]<<" "<<StaticForceMoment[5]<<endl;
	cout<<OtherHydrodynamic[0]<<" "<<OtherHydrodynamic[1]<<" "<<OtherHydrodynamic[2]<<" "<<OtherHydrodynamic[3]<<" "<<OtherHydrodynamic[4]<<" "<<OtherHydrodynamic[5]<<endl;
	cout<<EnvironForceMoment[0]<<" "<<EnvironForceMoment[1]<<" "<<EnvironForceMoment[2]<<" "<<EnvironForceMoment[3]<<" "<<EnvironForceMoment[4]<<" "<<EnvironForceMoment[5]<<endl<<endl;

    
	// ========== DEBUG START (调试代码，找到问题后可删除) ==========
    static int debug_counter = 0;
    debug_counter++;
    if (debug_counter % 10 == 0) { // 每10次调用打印一次，防止刷屏
        printf("\n--- TIME: %.3f ---\n", now_time);
        printf("Position Z: %.3f\n", model_states[2]); // 当前深度
        
        // 1. 检查浸没比例 (如果这里是 1.00，说明波浪判断有问题)
        printf("Wet Ratio: %.3f\n", wet_ratio); 
        
        // 2. 检查关键力的大小 (NED系，Z向下为正)
        // 重力 (应该约为 +9000 N)
        printf("Force Gravity (Z): %.3f\n", StaticForceMoment[2]); 
        
        // 波浪力+浮力 (如果完全出水，这里应该是 0)
        printf("Force Wave+Buoyancy (Z): %.3f\n", EnvironForceMoment[2]);
        
        // 阻尼力 (如果完全出水，这里应该接近 0)
        printf("Force Hydro Damping (Z): %.3f\n", OtherHydrodynamic[2]);
        
        // 总合力 (如果是正数，UUV应该加速下潜/掉落)
        printf("TOTAL FORCE Z: %.3f\n", temp_force[2]);
        
        printf("Velocity W: %.3f\n", w);
        printf("--------------------\n");
    }
    // ========== DEBUG END ==========
	
	return temp_force;
}

// //??6????????
// //??????
double (*Inverse(double Array[6][6]))[6]
{
	int m = 6;
	int n = 6;
	double array [2 * m + 1][2 * n + 1];
	for (int k = 0; k < 2 * m + 1; k++)  //?????????
	{
		for (int t = 0; t < 2 * n + 1; t++)
		{
				array[k][t] = 0.00000000;
		}
	}
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			array[i][j] = Array[i][j];
		}
	}

	for (int k = 0; k < m; k++){
		for (int t = n; t <= 2 * n; t++)
        {
			if ((t - k) == m){
				array[k][t] = 1.0;
			}
			else{
				array[k][t] = 0;
			}
		}
	}
	//????????
	for (int k = 0; k < m; k++){
		if (array[k][k] != 1){
			double bs = array[k][k];
			array[k][k] = 1;
			for (int p = k + 1; p < 2 * n; p++){
				array[k][p] /= bs;
			}
		}
		for (int q = 0; q < m; q++){
			if (q != k){
				double bs = array[q][k];
				for (int p = 0; p < 2 * n; p++){
					array[q][p] -= bs * array[k][p];
				}
			}
			else{
				continue;
			}
		}
	}
	static double NI[6][6];
	for (int x = 0; x < m; x++){
		for (int y = n; y < 2 * n; y++){
			NI[x][y - n] = array[x][y];
		}
	}
	return NI;
}

// //?????????
// //????
void matrixMultipy(double a[6][6] , double b[6], double c[6], int ra, int ca, int rb, int cb){
     if (ca != rb)
     {
         printf("Wrong Matrix Multiply\n");
     }
     for (int i = 0; i < ra; i++)
     {
		 double sum = 0;
		 for (int k = 0; k < ca; k++)
		 {
			 sum += a[i][k] * b[k];
		 }
		 c[i] = sum;
     }
 }
