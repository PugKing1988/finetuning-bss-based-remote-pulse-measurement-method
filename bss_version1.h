#ifndef BSS_VERSION1_H
#define BSS_VERSION1_H

#include <QtGui/QMainWindow>
#include "ui_bss_version1.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <QMainWindow>
#include <QThread>
#include <QPointF>
using namespace cv;
typedef unsigned char       BYTE;

class VideoWorker : public QObject {
    Q_OBJECT
 
public:
    VideoWorker ();
    ~VideoWorker ();
 
public slots:
    void process();
 
signals:
    void finished();
    void error(QString err);
	void frameCaptured();
private:
	
    // add your variables here
};

class Worker : public QObject {
    Q_OBJECT
 
public:
    Worker();
    ~Worker();
 std::vector<Rect> Worker::faceDec();
public slots:
    void process();
 
signals:
    void finished();
    void error(QString err);
 
private:
    // add your variables here
};
class QPointF3D :public QPointF
{
public:
	QPointF3D (): QPointF(){};
	QPointF3D (double _x, double _y, double _z): QPointF(_x,_y){lz=_z;};
	double z(){return lz;};
private:
	double lz;
};
class testOpencv : public QMainWindow
{
	Q_OBJECT

public:
	testOpencv(QWidget *parent = 0);
	~testOpencv();
public slots:
		//void timerEvent(QTimerEvent * e);
		//std::vector<Rect> faceDec(Mat frame);
		QPointF3D getAverage( std::vector<Rect> faces,Mat frame, int HStep,int VStep);
		void processFrame();
		
private:
	Ui::BSS_version1Class ui;
	int tm;
	int faceTimer;
	QImage qimgOriginal;
};

#endif // TESTOPENCV_H

