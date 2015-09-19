CFLAGS=-Wall -O0 -g `gsl-config --cflags` `pkg-config --cflags standard` `pkg-config --cflags glib-2.0` `pkg-config --cflags apr-1` `xml2-config --cflags` `pkg-config --cflags dpgmm`
ASFLAGS=-g `pkg-config --libs standard` `pkg-config --libs glib-2.0` `pkg-config --libs apr-1` `xml2-config --libs` 
CC=gcc
LOADLIBES=-lpthread -lm `gsl-config --libs` `pkg-config --libs standard`
metropolis: multinomial_logit.o
clean:
	rm *.o metropolis
