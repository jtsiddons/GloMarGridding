#!/usr/bin/env Rscript
my_args = commandArgs(trailingOnly=TRUE)
if ( length(my_args) == 0 ) {my_args<-my_args2}

ycell.want<-as.numeric(my_args[1])

library(ncdf4)
library(yaml)

source("./read_netR.R")

configx <- read_yaml(file = "vario_config.yml")
config <- configx$default

source_dir<-config$source_dir
tendeg_dir<-config$tendeg_dir
if(!dir.exists(tendeg_dir)) dir.create(tendeg_dir,recursive=T)

xgrid<-seq(-180, 175, 10)
ygrid<-seq( -90,  85, 10)

# full list of input files
flist<-list.files(source_dir,full.names=T,patt="nc")

# get dimensions etc.
tmp_nc<-read_netR(flist[1])
lon<-tmp_nc$dims$lon
lat<-tmp_nc$dims$lat
xcell<-findInterval(lon,xgrid)
ycell<-findInterval(lat,ygrid)
rm(tmp_nc)

for ( xcell.want in unique(xcell) ) {
    # processing per 10x10 box
    n.cell<-sum(xcell==xcell.want)
    start<-c(min(which(xcell==xcell.want)),min(which(ycell==ycell.want)),1)
    count<-c(n.cell,n.cell,1)
    lon.min<-floor(lon[start[1]])
    lat.min<-floor(lat[start[2]])
    cat("running data prep for lon/lat box:",lon.min,lat.min,"\n")
    outfile<-paste0(tendeg_dir,"data_",lon.min,"_",lat.min,".nc")
    if(file.exists(outfile) ) {
        cat("file exists, skipping \n")
        next
    }
  
    sst_anomaly<-array(NA,dim=c(n.cell,n.cell,length(flist)))
    sst_uncertainty<-array(NA,dim=c(n.cell,n.cell,length(flist)))
    sea_ice_fraction<-array(NA,dim=c(n.cell,n.cell,length(flist)))
    sea_fraction<-array(NA,dim=c(n.cell,n.cell,length(flist)))
    my.date<-rep(NA,times=length(flist))
    
    for( ii in 1:length(flist) ){
        cat(ii,flist[ii],"\n")
        # open & read netcdf file
        infile <- flist[ii]
        year <- substr(basename(infile),1,4)
        month <- substr(basename(infile),5,6)
        day <- substr(basename(infile),7,8)
        my.date[ii]<-as.Date(paste0(year,"-",month,"-",day))
        ncin <- nc_open(infile)
        sst_anomaly[,,ii] <- ncvar_get(ncin,'sst_anomaly', start = start , count = count )
        sst_uncertainty[,,ii] <- ncvar_get(ncin,'sst_uncertainty', start = start , count = count )
        sea_ice_fraction[,,ii] <- ncvar_get(ncin,'sea_ice_fraction', start = start , count = count )
        sea_fraction[,,ii] <- ncvar_get(ncin,'sea_fraction', start = start , count = count )
        nc_close(ncin)
     }

     numdate<-as.numeric(my.date)
     xdim<-ncdim_def("lon", "degreeE", 
                     lon[xcell==xcell.want], 
                     unlim=FALSE, 
                     create_dimvar=TRUE, 
                     calendar=NA, 
                     longname="longitude" )
     ydim<-ncdim_def("lat", "degreeN", 
                     lat[ycell==ycell.want], 
                     unlim=FALSE, 
                     create_dimvar=TRUE, 
                     calendar=NA, 
                     longname="latitude" )
     tdim<-ncdim_def("time", "days since 1970-01-01", 
                     numdate, 
                     unlim=FALSE, 
                     create_dimvar=TRUE, 
                     calendar=NA, 
                     longname="days since 1970-01-01" )
     var_sst_anomaly <- ncvar_def("sst_anomaly", "K", 
                                  list(xdim,ydim,tdim), 
                                  NA, 
                                  longname="sst_anomaly", 
                                  prec="float")
     var_sst_uncertainty <- ncvar_def("sst_uncertainty", "K", 
                                      list(xdim,ydim,tdim), 
                                      NA, 
                                      longname="sst_uncertainty", 
                                      prec="float")
     var_sea_ice_fraction <- ncvar_def("sea_ice_fraction", "K", 
                                       list(xdim,ydim,tdim), 
                                       NA, 
                                       longname="sea_ice_fraction", 
                                       prec="float")
     var_sea_fraction <- ncvar_def("sea_fraction", "K", 
                                   list(xdim,ydim,tdim), 
                                   NA, 
                                   longname="sea_fraction", 
                                   prec="float")
     nc<-nc_create( outfile, var_sst_anomaly, force_v4=FALSE, verbose=FALSE )
     ncvar_put( nc, "sst_anomaly", sst_anomaly)
     nc<-ncvar_add( nc, var_sst_uncertainty, verbose=FALSE, indefine=FALSE )
     ncvar_put( nc, "sst_uncertainty", sst_uncertainty)
     nc<-ncvar_add( nc, var_sea_ice_fraction, verbose=FALSE, indefine=FALSE )
     ncvar_put( nc, "sea_ice_fraction", sea_ice_fraction)
     nc<-ncvar_add( nc, var_sea_fraction, verbose=FALSE, indefine=FALSE )
     ncvar_put( nc, "sea_fraction", sea_fraction)
     nc_close( nc )
 }  # end x-loop over 10 degree cells
