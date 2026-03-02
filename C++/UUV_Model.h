/*
 * UUV_Model.h
 *
 *  Created on: 2021-2-3
 *      Author: Lenovo
 */

#ifndef UUV_MODEL_H_
#define UUV_MODEL_H_
#define AOFFSET_DATA 4
#define SIZE_CHECKSUM_ENDFLAG 4
#define OCEAN_CURRENT_FILE_NAME "/uuv/cfg/motion/ocean_current.json"

typedef struct{
    double minValue, maxValue;
    double constTime;
    double maxRate;
    double Cn;
}DRIVERINFO;

typedef struct _SensorMSG_{
	double  u,v,w;                   	 // m/s
	double  p,q,r;						 // degree/s
	double  roll,pitch,heading;			 // degree
	double  north,east;              	 // m
	double  depth;                   	 // m
	double  lon,lat;                 	 // degree
}SensorMSG;

typedef struct _OceanState_{
	double  uw,vw,ww;             	 
}Ocean_V_State;

#pragma pack(1)
typedef struct StructDate
{
	unsigned short year;
	unsigned char month;
	unsigned char date;
	unsigned char hour;
	unsigned char minute;
	unsigned char second;
}structDate_t;
#pragma pack()

#pragma pack(1)
typedef struct UuvInf
{
	unsigned char ID;
	double Lon;
	double Lat; 
	double V; 
	double C;
	double H;
	double Roll; 
	double Pitch;
	double Yaw;
	
}struct_UuvInf_t;
#pragma pack()

#pragma pack(1)
typedef struct BALANCE_DEVICE_STATUS_info_QUERY_RSP
{
	unsigned short BusVoltage;
	short QaxisCurrent;
	unsigned short Cmdvolume;
	short Fbkvolume; //容积反馈
	unsigned short Volumerate;
	short MotorSpeed;
	unsigned char OilDoughSidePressure;
	unsigned char TankSidePressure;
	unsigned char PumpOutletPressure;
	unsigned char WorkingState;
	unsigned char Health;
}structBALANCE_DEVICE_STATUS_info_QUERY_RSP_t;

typedef struct _OceanConFiguration_ 
{
	double Ocean_UC;
	double Ocean_VC;
}OCEANCONFIGURATION;

#pragma pack()


#endif /* UUV_MODEL_H_ */
