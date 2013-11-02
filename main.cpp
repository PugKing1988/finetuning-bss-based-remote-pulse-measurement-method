#include "bss_version1.h"
#include <QtGui/QApplication>
#include <cv.h>
#include <Windows.h>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	testOpencv w;
	w.show();
	return a.exec();
}
