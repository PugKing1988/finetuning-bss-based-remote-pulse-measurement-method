#include "bss_version1.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>//
#include "opencv2/gpu/gpu.hpp" 
#include <opencv2/objdetect/objdetect.hpp>//
#include <cv.h>
#include "fft.h"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <windows.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <Qtime>
#include <QDebug>
#include <QMutex>
#include <cstdlib>
#include <opencv2/legacy/legacy.hpp>
VideoCapture capture("D:\\Tester\\Video\\Sam_11_44_pm_1.avi");


#define FFTSIZE 256
#define FFTN 8
#define WORK_FPS 30
#define M_PI 3.14159265358979323846
using namespace std;
using namespace cv;
using namespace cv::gpu;
VideoCapture cap;
//VideoCapture capture;
QMap <double,double> doo;
QMap <double,double> too;
QList <double> rHist;
QList <double> gHist;
QList <double> bHist;
double averager = 0;
int freqIndex = 0;
int freqIndex32 = 0;
int freqSecIndex = 0;
int freqThirdIndex = 0;
double inX[1024];
double inY[1024];
double higher= 0;
//int frmcount;
bool flag2 = false;
bool flag = true;
double secFreqWant = 0;
double thirdFreqWant = 0;
Mat frame ;
Mat threadFrame;
int fps =0;
int proc_fps=0;
int procframe=0;
int lastsec=0;
int fCounter=0;
double highFreq=0;
Mat globalImageCopy;
double slowFreq = 0;
int counter=0;
std::vector<Rect> globalFaces;
bool initialised ;
QMutex locker;
int itemCount;
double nextHigh=0;
double nextBest= 0;
bool running;
int lastd = 0;
QTime t;
double pred;
int timeCount;
int timeCounter;
int s; 
double lastheartrate;
double lasthr;
QList<double> heart;
vector <double> comp;
double predi;
double rMean,gMean,bMean;
double rStd,gStd,bStd;
double  oldr=0;
double  oldg=0;
double  oldb=0;
double yb,yg,yr;
double psh;
const float A[] = { 1, 1, 0, 1 };
CvKalman* kalman;
CvMat* state = NULL;
CvMat* measurement;
const CvMat* prediction;

int lastg = 0;
QList <double> noo;
double hr ;
double fhr;
#define PI 3.14159265

void kalman_filter(float FoE_x, float prev_x)
{
	if (_isnan(prev_x) ||_isnan(FoE_x))
		return;
    prediction = cvKalmanPredict(kalman, 0 );
   // printf("KALMAN: %f %f %f\n" , prev_x, prediction->data.fl[0] , prediction->data.fl[1] );
	
    measurement->data.fl[0] = FoE_x;
	//cvMatMulAdd( kalman->measurement_matrix, state, measurement, measurement );
    cvKalmanCorrect( kalman, measurement);
	//qDebug() << measurement->data.fl[0] ;

}


void detrend(double array[], int n)
{
     double x, y, a, b;
     double sy = 0.0,
            sxy = 0.0,
            sxx = 0.0;
     int i;

     for (i=0, x=(-n/2.0+0.5); i<n; i++, x+=1.0)
     {
          y = array[i];
          sy += y;
          sxy += x * y;
          sxx += x * x;
     }
     b = sxy / sxx;
     a = sy / n;

     for (i=0, x=(-n/2.0+0.5); i<n; i++, x+=1.0)
          array[i] -= (a+b*x);
}




VideoWorker::VideoWorker() {
    // you could copy data from constructor arguments to internal variables here.
}
 
// --- DECONSTRUCTOR ---
VideoWorker::~VideoWorker() {
    // free resources
}
 
// --- PROCESS ---
// Start processing data.
void VideoWorker::process() {
	
    // allocate resources using new here
    double work_fps ;
	int elapsed=0;
	
	//t.start();
	//qDebug() << "123";
	lastsec=0;
	while(running)
	{ 
		//elapsed=t.elapsed();
		//t.restart();
		counter++;
		s=QTime::currentTime().second();
		if (s!=lastsec)
		{
			//qDebug() <<counter ;;
			fps=counter;
			proc_fps=procframe;
			procframe=0;
			counter=0;
		}
		lastsec=s;
		//qDebug() << "321";
		//get consistent 30-fps video
		cap>>frame;
		//capture >>frame;
		if(frame.empty())
			exit(0);
		locker.lock();
		frame.copyTo(threadFrame);
		locker.unlock();
		//qDebug() << "dog123";
		emit frameCaptured();
		initialised=true;
		Sleep(41.6);
	}
	
    emit finished();
}


// --- CONSTRUCTOR ---
Worker::Worker() {
    // you could copy data from constructor arguments to internal variables here.
}
 
// --- DECONSTRUCTOR ---
Worker::~Worker() {
    // free resources
}
 
// --- PROCESS ---
// Start processing data.
void Worker::process() {
    // allocate resources using new here
    faceDec();
    emit finished();
}
std::vector<Rect> Worker::faceDec(){
	int framecount=0;
	std::vector<Rect> faces;
	Mat grayscaleFrame;
    CascadeClassifier face_cascade;
	face_cascade.load("C:\\OpenCV245\\data\\haarcascades\\haarcascade_frontalface_alt.xml");
    while(flag&& running)
    {
		if (!initialised){Sleep(1000);continue;}
		framecount++;
		locker.lock();
        cvtColor(globalImageCopy, grayscaleFrame, CV_BGR2GRAY);
		locker.unlock();
        equalizeHist(grayscaleFrame, grayscaleFrame);
		face_cascade.detectMultiScale(grayscaleFrame, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, Size(10,10));
		locker.lock();
		globalFaces=faces;
		locker.unlock();
		if (faces.size())
	
			Sleep(5000);
		else
			Sleep(100);
    }
	
	return faces;
}

extern "C"
{


#include "../JnS-1.2/Matutil.h" 
#include "../JnS-1.2/JnS.h"
};


testOpencv::testOpencv(QWidget *parent)
	: QMainWindow(parent)
{
	kalman = cvCreateKalman( 1, 1 );
	state = cvCreateMat( 1, 1, CV_32FC1 );
	measurement = cvCreateMat( 1, 1, CV_32FC1 );
	cvSetIdentity( kalman->measurement_matrix,cvRealScalar(1) ); // H
	memcpy( kalman->transition_matrix->data.fl, A, sizeof(A));
	cvSetIdentity( kalman->process_noise_cov, cvRealScalar(0.1));// Q
	cvSetIdentity(kalman->measurement_noise_cov, cvRealScalar(10));//R
	cvSetIdentity( kalman->error_cov_post, cvRealScalar(100));
	kalman->state_post->data.fl[0] = 60;
	kalman->state_pre->data.fl[0] =60;
	yb = yg = yr = 0;
	rMean = gMean = bMean = 0 ;
	
	rStd = gStd = bStd = 0;

	running=true;
	ui.setupUi(this);
	//namedWindow( "videoOut", 0 );// Create a window for display.
	cap.open(0);
	Sleep(300);
	QThread* thread = new QThread();
	VideoWorker* worker = new VideoWorker();
	worker->moveToThread(thread);
	connect(worker, SIGNAL(error(QString)), this, SLOT(errorString(QString)));
	connect(thread, SIGNAL(started()), worker, SLOT(process()));
	connect(worker, SIGNAL(finished()), thread, SLOT(quit()));
	connect(worker, SIGNAL(finished()), worker, SLOT(deleteLater()));
	connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));
	connect (worker, SIGNAL (frameCaptured()),this,SLOT(processFrame()));
	thread->setPriority(QThread::TimeCriticalPriority);
	int p=thread->priority();
	thread->start();
	thread->setPriority(QThread::TimeCriticalPriority);
	p=thread->priority();
	QThread* threadface = new QThread();
	Worker* workerface = new Worker();
	workerface->moveToThread(threadface);
	connect(workerface, SIGNAL(error(QString)), this, SLOT(errorString(QString)));
	connect(threadface, SIGNAL(started()), workerface, SLOT(process()));
	connect(workerface, SIGNAL(finished()), threadface, SLOT(quit()));
	connect(worker, SIGNAL(finished()), workerface, SLOT(deleteLater()));
	connect(threadface, SIGNAL(finished()), threadface, SLOT(deleteLater()));
	threadface->start();
	
	return;

}

testOpencv::~testOpencv()
{
	running=false;
	cap.release();
	
}
QPointF3D testOpencv::getAverage( vector<Rect> faces,Mat frame, int Hstep, int VStep)
{
	double r,g,b;
	double R,G,B;
	R=G=B =0;
	r=g=b=0;
	int bcounter=0;
	unsigned char * aaa= frame.data;
	
	for(int i = 0; i < faces.size(); i++)
	{
		for (int j=faces[i].x+0.05*faces[i].width;j<(faces[i].x + 0.95*faces[i].width);j++)
		{
			for (int k=faces[i].y+0.05*faces[i].height;k<(faces[i].y + 0.95*faces[i].height);k++)
			{
				bcounter++;
				b=*(aaa+j*Hstep+k*VStep);
				g=*(aaa+j*Hstep+k*VStep+1);
				r=*(aaa+j*Hstep+k*VStep+2);
				 if (( r > 95) && ( g > 40 ) && ( b > 20 ) &&
						   (max(r, max( g,b) ) - min(r, min(g, b) ) > 15) &&
									 (std::abs(r - g) > 15) && (r > g) && (r > b))
				{
					B+=*(aaa+j*Hstep+k*VStep);
					G+=*(aaa+j*Hstep+k*VStep+1);
					R+=*(aaa+j*Hstep+k*VStep+2);/*
					*(aaa+j*hstep+k*vstep) = 255;
					*(aaa+j*hstep+k*vstep+1) = 255;
					*(aaa+j*hstep+k*vstep+2) = 255;*/


				 }
			}
		}
	}	
	R/=bcounter;
	G/=bcounter;
	B/=bcounter;
	return QPointF3D(R,G,B);
	//r/=bcounter;
	//g/=bcounter;
	//b/=bcounter;
	//return QPointF3D(r,g,b);
}
void testOpencv::processFrame()
{
	//return; 
	//counter++;proc_fps
	procframe++;
	Mat outFrame = Mat(320,240,CV_8UC3);
	frame.copyTo(outFrame);

	locker.lock();
	globalImageCopy=threadFrame.clone();
	locker.unlock();
	char text[100];
	double val = cap.get(CV_CAP_PROP_FRAME_COUNT);
	sprintf(text,  "input Fps %i, proc %i", fps ,proc_fps);
	CvFont font;
	putText(outFrame, text, cvPoint(300,100),FONT_HERSHEY_COMPLEX_SMALL,1.2, cvScalar(150,0,0), 1, CV_AA);
	
	std::vector<Rect> go;
	locker.lock();
	go=globalFaces; //pick it up from global thread
	locker.unlock();
	int i = 0;
	IplImage outImage(outFrame);

	
	QList <double> datapoints1;
	QList <double> datapoints2;
	QList <double> datapoints3;
	QList <double> datapoints;
	double maxFreq = 0;
	double myMaxFreq = 0;
	double compareFreq = 0;
	double myBestFreq = 0;
	double myBestMag = 0;
	double sum1,sum2;
	sum1 = sum2 = 0;
	double __output[3*3];
	double __input[FFTSIZE*3];  
	double __rinput[FFTSIZE];
	double __ginput[FFTSIZE];
	double __binput[FFTSIZE];
	/*
	int num_device = getCudaEnabledDeviceCount();
	qDebug() << num_device;
	qDebug() << getDevice();*/
	if(go.size() > 0){
		fCounter++;
		rectangle(outFrame, Point(go[0].x,go[0].y), Point(go[0].x+go[0].width,go[0].y+go[0].height), CV_RGB(0,255,0), 2, 8, 0);
	//	rectangle(outFrame, Point(go[0].x+go[0].width*.25,go[0].y+go[0].height*.25), Point(go[0].x+go[0].width*.75,go[0].y+go[0].height*.75), CV_RGB(255,0,0), 1, 8, 0);
		QPointF3D avg=getAverage(go,outFrame,outImage.nChannels,outImage.widthStep);
		//smooth it just a little bit
	if (rHist.size())
	{
		rHist.append(rHist.last()-(rHist.last()-avg.x())/5.0);
		gHist.append(gHist.last()-(gHist.last()-avg.y())/5.0);
		bHist.append(bHist.last()-(bHist.last()-avg.z())/5.0);
	}
	else
	{
		rHist.append(avg.x());
		gHist.append(avg.y());
		bHist.append(avg.z());
	}

		//remove excess datapoints
		while( rHist.size()>FFTSIZE)//w*0.5)
			rHist.removeFirst();
		while( gHist.size()>FFTSIZE)//w*0.5)
			gHist.removeFirst();
		while( bHist.size()>FFTSIZE)//w*0.5)
			bHist.removeFirst();
		if (rHist.size()<1.33*fps)
		{/*tm=startTimer(10);*/	return;}
		//qDebug() << rHist.size();


		gMean=gHist[0];
		rMean=rHist[0];
		bMean=bHist[0];
		

		//zero mean centre
		for (int i=0;i<rHist.size()-1;i++)
			rMean+=rHist[i+1];
		rMean/=rHist.size();
		for (int i=0;i<rHist.size();i++)
			rStd+=(rHist[i]-rMean)*(rHist[i]-rMean);
		rStd=sqrt(rStd/rHist.size());
		if (rStd==0)rStd=1;
		
		for (int i=0;i<gHist.size()-1;i++)
			gMean+=gHist[i+1];
		gMean/=gHist.size();
		for (int i=0;i<gHist.size()-1;i++)
			gStd+=(gHist[i+1]-gMean)*(gHist[i+1]-gMean);
		gStd=sqrt(gStd/gHist.size());
		if (gStd==0)gStd=1;
		
		for (int i=0;i<bHist.size()-1;i++)
			bMean+=bHist[i+1];
		bMean/=bHist.size();
		for (int i=0;i<gHist.size()-1;i++)
			bStd+=(bHist[i+1]-bMean)*(bHist[i+1]-bMean);
		bStd=sqrt(bStd/bHist.size());
		if (bStd==0)bStd=1;



		
		
		//draw raw data points
		int drawoffset=40;
		for (int i=0;i<rHist.size()-1;i++)
		{
			line(outFrame,Point(i,drawoffset+(rHist[i]-rMean)*5.0/rStd),Point(i+1,drawoffset+(rHist[i+1]-rMean)*5.0/rStd),CV_RGB(200,0,0));
		}
		drawoffset=70;//-gMean;
		for (int i=0;i<gHist.size()-1;i++)
		{
			line(outFrame,Point(i,drawoffset+(gHist[i]-gMean)*5.0/gStd),Point(i+1,drawoffset+(gHist[i+1]-gMean)*5.0/gStd),CV_RGB(0,200,0));
			
		}
		drawoffset=100;//-bMean;
		for (int i=0;i<bHist.size()-1;i++)
		{
			line(outFrame,Point(i,drawoffset+(bHist[i]-bMean)*5.0/bStd),Point(i+1,drawoffset+(bHist[i+1]-bMean)*5.0/bStd),CV_RGB(0,0,200));
			
		}




		itemCount=rHist.size();
		//normalize the raw data - could detrend it here also
		for (int i=0;i<itemCount;i++)
		{
			__input[i]=(rHist[i]-rMean)/rStd;
			__input[itemCount+i]=(gHist[i]-gMean)/gStd;
			__input[itemCount*2+i]=(bHist[i]-bMean)/bStd;
			__rinput[i] = __input[i];
			__ginput[i] = __input[itemCount+i];
			__binput[i] = __input[itemCount*2+i];

		}
		//detrending here
		detrend(__rinput,itemCount);
		detrend(__ginput,itemCount);
		detrend(__binput,itemCount);
		for (int i=0;i<itemCount;i++)
		{
			__input[i]=__rinput[i];
			__input[itemCount+i]=__ginput[i];
			__input[itemCount*2+i]=__binput[i];

		}

		///////////////////////////Joint Approximate diagonalization of eigen matrices/////////////////
		Jade(__output,__input,3,rHist.size());
		if (_isnan(__input[0]))
			return;

		//for (int i=0;i<itemCount;i++)
		//{
		///*	__input[i]=(rHist[i]-rMean)/rStd;
		//	__input[itemCount+i]=(gHist[i]-gMean)/gStd;
		//	__input[itemCount*2+i]=(bHist[i]-bMean)/bStd;*/
		//	__rinput[i] = __input[i];
		//	__ginput[i] = __input[itemCount+i];
		//	__binput[i] = __input[itemCount*2+i];

		//}
		//detrend(__rinput,itemCount);
		//detrend(__ginput,itemCount);
		//detrend(__binput,itemCount);
		//for (int i=0;i<itemCount;i++)
		//{
		//	__input[i]=__rinput[i];
		//	__input[itemCount+i]=__ginput[i];
		//	__input[itemCount*2+i]=__binput[i];

		//}
		for (int i=0;i<itemCount;i++)
		{
			double r=__input[i]*20;
			double g=__input[itemCount+i]*20;
			double b=__input[itemCount*2+i]*20;
			datapoints1.push_back(r/2.0);
			datapoints2.push_back(g/2.0);
			datapoints3.push_back(b/2.0);
			if (yr==0 ||_isnan(yr))yr=r;else yr-=(yr-r)/3.0;//why divied by 3
			if (yg==0||_isnan(yg))yg=g;else yg-=(yg-g)/3.0;
			if (yb==0||_isnan(yb))yb=b;else yb-=(yb-b)/3.0;
			line(outFrame,Point(i,150+oldr),Point(i+1,150+yr),CV_RGB(255,0,0));
			line(outFrame,Point(i,200+oldg),Point(i+1,200+yg),CV_RGB(0,255,0));
			line(outFrame,Point(i,250+oldb),Point(i+1,250+yb),CV_RGB(0,0,255));
			oldb=yb;oldg=yg;oldr=yr;
		}



		memset(inX,0,1024*sizeof(double));
		memset(inY,0,1024*sizeof(double));

		for (int i=0;i<datapoints1.size();i++)
		{
			inX[i]=datapoints1[i];//i;
			inY[i]=0;
		}

		//do fft on red channel
		//memset(inX,0,1024*sizeof(double));
		//memset(inY,0,1024*sizeof(double));
		

		

		FFT(1,FFTN,inX,inY);
		

		for (int i=1;i<datapoints1.size()/2;i++)
		{
			double v1=sqrt(inX[i]*inX[i]+inY[i]*inY[i]);
			double v2=sqrt(inX[i+1]*inX[i+1]+inY[i+1]*inY[i+1]);
			line(outFrame,Point(i*4,300-v1*10),Point((i+1)*4,300-v2*10),CV_RGB(255,0,0));
		}

		for(int i =1;i<FFTSIZE;i++){
			
			double freq = (double)i*fps/FFTSIZE;
			if (freq > 4 || freq < 0.75){
				inX[i] = 0;
				inY[i] = 0;
			}
			
		}

		maxFreq = 0;
		for(int i = 1;i< FFTSIZE;i++ ){
			double v1 = sqrt(inX[i]*inX[i]+inY[i]*inY[i]);
			if(maxFreq < v1){
				maxFreq = v1;

				freqIndex = i;
			}
		}
		int i =freqIndex;
		
		sum1 =	(i-1)*sqrt(inX[i-1]*inX[i-1]+inY[i-1]*inY[i-1])+
						(i)*sqrt(inX[i]*inX[i]+inY[i]*inY[i])+
						(i+1)*sqrt(inX[i+1]*inX[i+1]+inY[i+1]*inY[i+1]);
		sum2 =	sqrt(inX[i-1]*inX[i-1]+inY[i-1]*inY[i-1])+
						sqrt(inX[i]*inX[i]+inY[i]*inY[i])+
						sqrt(inX[i+1]*inX[i+1]+inY[i+1]*inY[i+1]);
	
		if (sum2)
			myMaxFreq=sum1/sum2;
		int g = QTime::currentTime().second();
		if(g == lastg)
		{
			//doo.insert(maxFreq,freqIndex);
			//doo.insert(maxFreq,myMaxFreq);
			doo.insert(myMaxFreq,maxFreq);
		}
		myBestFreq=myMaxFreq;
		myBestMag=maxFreq;

		line(outFrame,Point(myMaxFreq*4,280),Point(myMaxFreq*4,320),CV_RGB(255,0,0));
		double freqWant = myMaxFreq*fps/FFTSIZE*60.0;
		char text[100];
		sprintf(text,  "Heart rate: %2.2f=%2.2f=%2.2f ",  myMaxFreq,maxFreq,freqWant);
		//sprintf(text,  "Heart rate: %2.2f=%2.2f ",  myMaxFreq,freqWant);
		CvFont font;
		putText(outFrame, text, cvPoint(230,350),CV_FONT_HERSHEY_SIMPLEX,0.8, cvScalar(0,0,255), 2, CV_AA);
		
		////////////////////////////////////////green second component


		for (int i=0;i<datapoints2.size();i++)
		{
			inX[i]=datapoints2[i];
			inY[i]=0;
		}


		FFT(1,FFTN,inX,inY);
		for (int i=1;i<datapoints2.size()/2;i++)
		{
			double v1=sqrt(inX[i]*inX[i]+inY[i]*inY[i]);
			double v2=sqrt(inX[i+1]*inX[i+1]+inY[i+1]*inY[i+1]);	
			line(outFrame,Point(i*4,350-v1*10),Point((i+1)*4,350-v2*10),CV_RGB(0,255,0));  //DRAW THE FFT
		
			
		}
	
		for(int i =1;i<FFTSIZE;i++){
			double v1 = sqrt(inX[i]*inX[i]+inY[i]*inY[i]);
			double freq = (double)i*fps/FFTSIZE;
			if (freq > 4 || freq < 0.75){
				inX[i] = 0;
				inY[i] = 0;
			}

		}
		maxFreq = 0;
		
		for(int i = 1;i< FFTSIZE;i++ ){
			double v1 = sqrt(inX[i]*inX[i]+inY[i]*inY[i]);
			double v2= sqrt(inX[i+1]*inX[i+1]+inY[i+1]*inY[i+1]);
			if(maxFreq < v1){
				maxFreq = v1;
				freqSecIndex = i;
			}
		}
		i = freqSecIndex;
		
		sum1 =	(i-1)*sqrt(inX[i-1]*inX[i-1]+inY[i-1]*inY[i-1])+
						(i)*sqrt(inX[i]*inX[i]+inY[i]*inY[i])+
						(i+1)*sqrt(inX[i+1]*inX[i+1]+inY[i+1]*inY[i+1]);
		sum2 =	sqrt(inX[i-1]*inX[i-1]+inY[i-1]*inY[i-1])+
						sqrt(inX[i]*inX[i]+inY[i]*inY[i])+
						sqrt(inX[i+1]*inX[i+1]+inY[i+1]*inY[i+1]);
			
		if (sum2)
			myMaxFreq=sum1/sum2;
		if(g == lastg)
		{
			//doo.insert(maxFreq,freqIndex);
			//doo.insert(maxFreq,myMaxFreq);
			doo.insert(myMaxFreq,maxFreq);
		}
		if (maxFreq >myBestMag)
		{
			myBestMag=maxFreq;
			myBestFreq=myMaxFreq;
		}

		line(outFrame,Point(myMaxFreq*4,330),Point(myMaxFreq*4,370),CV_RGB(0,255,0));

		

		secFreqWant = myMaxFreq*fps/FFTSIZE*60.0;
		if (freqWant > secFreqWant){
			highFreq = freqWant;}else{
				highFreq = secFreqWant;}

		sprintf(text,  "Heart rate: %2.2f=%2.2f=%2.2f ",  myMaxFreq,maxFreq,secFreqWant);
		putText(outFrame, text, cvPoint(230,400),CV_FONT_HERSHEY_SIMPLEX,0.8, cvScalar(0,255,0), 2, CV_AA);
	

		////////////////////////////////////////blue

		for (int i=0;i<datapoints3.size();i++)
		{
			inX[i]=datapoints3[i];
			inY[i]=0;
		}
	

		FFT(1,FFTN,inX,inY);
		for (int i=1;i<datapoints3.size()/2;i++)
		{
			double v1=sqrt(inX[i]*inX[i]+inY[i]*inY[i]);
			double v2=sqrt(inX[i+1]*inX[i+1]+inY[i+1]*inY[i+1]);	
		line(outFrame,Point(i*4,400-v1*10),Point((i+1)*4,400-v2*10),CV_RGB(0,0,255));
	
		}
		for(int i =1;i<FFTSIZE;i++){
			double v1 = sqrt(inX[i]*inX[i]+inY[i]*inY[i]);
			double freq = (double)i*fps/FFTSIZE;
			if (freq > 4 || freq < 0.75){
				inX[i] = 0;
				inY[i] = 0;
			}

		}
		 maxFreq = 0;
		 for(int i = 1;i< FFTSIZE;i++ ){
			double v1 = sqrt(inX[i]*inX[i]+inY[i]*inY[i]);
			double v2= sqrt(inX[i+1]*inX[i+1]+inY[i+1]*inY[i+1]);
			if(maxFreq < v1){
				maxFreq = v1;
				freqThirdIndex = i;
			}
		}
		int idx =freqThirdIndex;
		
		sum1 =	(idx-1)*sqrt(inX[idx-1]*inX[idx-1]+inY[idx-1]*inY[idx-1])+
						(idx)*sqrt(inX[idx]*inX[idx]+inY[idx]*inY[idx])+
						(idx+1)*sqrt(inX[idx+1]*inX[idx+1]+inY[idx+1]*inY[idx+1]);
		sum2 =	sqrt(inX[idx-1]*inX[idx-1]+inY[idx-1]*inY[idx-1])+
						sqrt(inX[idx]*inX[idx]+inY[idx]*inY[idx])+
						sqrt(inX[idx+1]*inX[idx+1]+inY[idx+1]*inY[idx+1]);
		if (sum2)
			myMaxFreq=sum1/sum2;
		if(g == lastg)
		{
			//doo.insert(maxFreq,freqIndex);
			//doo.insert(maxFreq,myMaxFreq);
			doo.insert(myMaxFreq,maxFreq);
		}
		if (maxFreq >myBestMag)
		{
			myBestMag=maxFreq;
			myBestFreq=myMaxFreq;
		}
		line(outFrame,Point(myMaxFreq*4,380),Point(myMaxFreq*4,420),CV_RGB(0,0,255));
		thirdFreqWant = myMaxFreq*fps/FFTSIZE*60.0;
		double bestFreWant = myBestFreq*fps/FFTSIZE*60.0;
		sprintf(text,  "Heart rate: %2.2f=%2.2f=%2.2f ",  myMaxFreq,maxFreq,thirdFreqWant);
		//sprintf(text,  "Heart rate: %2.2f=%2.2f ",  myMaxFreq,thirdFreqWant);
		putText(outFrame, text, cvPoint(230,450),CV_FONT_HERSHEY_SIMPLEX,0.8, cvScalar(255,0,0), 2, CV_AA);
		if(highFreq < thirdFreqWant){
			highFreq = thirdFreqWant;
		}
		
		
		if (lasthr==0)
			lasthr=fhr;
		if(g != lastg)
		{
			
			if(hr!=0){
				kalman_filter(lasthr,fhr);
				pred = prediction->data.fl[0];
			
			}
			QMap<double, double>::const_iterator i = doo.constEnd()-1;
			
			while (i != doo.constEnd()) {

				 too.insert(i.value(),i.key());
				 ++i;
			 }
			QMap<double, double>::const_iterator j = too.constEnd()-1;
		

			
			double ooo =j.value();
			fhr = ooo*fps/FFTSIZE*60.0;
		
			if(timeCounter<20 || (hr == 0)){
				if(j.key()>3 && fhr > 50)
					hr = ooo*fps/FFTSIZE*60.0;
			}else if(timeCounter>=20 || (hr !=0)){
				if((j.key()>3 && (abs(hr-fhr) < 12.00)) || (j.key() <3 && abs(hr-fhr)<6)){
							hr = fhr;
					}
				
			
			}

			qDebug() << timeCounter <<j.key() << fhr << hr <<pred;
			lasthr = fhr;
			timeCounter++;
			doo.clear();
			too.clear();
		}

		
		lastg = g;
		
	}	

		if(flag2 == true){
				sprintf(text,  "Please stay still");
				putText(outFrame, text, cvPoint(300,300),CV_FONT_HERSHEY_SIMPLEX,1, cvScalar(255,255,255), 2, CV_AA);
				sprintf(text,  "Your last successful HR was: %d", int(psh));
				putText(outFrame, text, cvPoint(100,50),CV_FONT_HERSHEY_SIMPLEX,0.8, cvScalar(0,0,0), 2, CV_AA);

		}else{
				
				sprintf(text,  "Best_HR: %d ",(int)higher);
				putText(outFrame, text, cvPoint(300,300),CV_FONT_HERSHEY_SIMPLEX,1.2, cvScalar(255,255,255), 2, CV_AA);
				sprintf(text,  "HR: %d ",(int)hr);
				putText(outFrame, text, cvPoint(300,200),CV_FONT_HERSHEY_SIMPLEX,1.2, cvScalar(255,255,255), 2, CV_AA);
				sprintf(text,  "fHR: %d ",(int)fhr);
				putText(outFrame, text, cvPoint(300,150),CV_FONT_HERSHEY_SIMPLEX,1.2, cvScalar(0,255,255), 2, CV_AA);
			//	sprintf(text,  "Your last successful HR was: %d", int(psh));
			//	putText(outFrame, text, cvPoint(100,50),CV_FONT_HERSHEY_SIMPLEX,0.8, cvScalar(0,0,0), 2, CV_AA);
			//	sprintf(text,  "Pred_HR: %d ",(int)predi);
			//	putText(outFrame, text, cvPoint(10,300),CV_FONT_HERSHEY_SIMPLEX,1.2, cvScalar(255,255,255), 2, CV_AA);
		}

		
		sprintf(text,  "%d",timeCounter);
		putText(outFrame, text, cvPoint(580,40),CV_FONT_HERSHEY_SIMPLEX,1, cvScalar(0,0,0), 2, CV_AA);

		this->setFixedSize(this->size());
		cvtColor(outFrame,outFrame,CV_BGR2RGB);
		QImage qimgOriginal((uchar*)outFrame.data,outFrame.cols,outFrame.rows,outFrame.step,QImage::Format_RGB888);
		ui.label->setPixmap(QPixmap::fromImage(qimgOriginal));
		//imshow("videoOut",outFrame);
	//	flag = true;

}

