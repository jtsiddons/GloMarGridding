read_netR <- function(fn) {

if (!("package:ncdf4" %in% search())) require(ncdf4)

  nc<-nc_open(fn)

  dnames_full<-names(nc$dim)
  dnames<-names(nc$dim)
  dnames<-dnames[dnames!="bounds"]
  dnames<-dnames[dnames!="bounds2"]
  dnames<-dnames[dnames!="bnds"]
  dnames<-dnames[dnames!="nv"]
  dnames<-dnames[dnames!="time_bnds"]
  dnames<-dnames[dnames!="nbnds"]

  #dd<- vector(mode = "list", length = nc$ndim)
  dd<- vector(mode = "list", length = length(dnames))

  #for ( ii in 1:nc$ndim ) {
  icount<-0
  for ( ii in which(dnames_full %in% dnames) ) {
   if ( !nc$dim[ii][dnames_full[ii]][[dnames_full[ii]]]$unlim ) {
   #if ( nc$dim[ii][dnames_full[ii]][[dnames_full[ii]]]$create_dimvar ) {
     icount<-icount+1
     dd[[icount]]<-ncvar_get(nc,dnames_full[ii])
   } else {
     icount<-icount+1
     dd[[icount]]<-seq(1:nc$dim[ii][dnames_full[ii]][[dnames_full[ii]]]$len)
   }
  }
  names(dd)<-dnames

  vv<- vector(mode = "list", length = nc$nvars)
  vnames<-rep(NA,times=nc$nvars)
  for ( ii in 1:nc$nvars ) { vnames[ii] <- nc$var[ii][[1]]$name }
  vunits<-rep(NA,times=nc$nvars)
  for ( ii in 1:nc$nvars ) { vunits[ii] <- nc$var[ii][[1]]$units }

  for ( ii in 1:nc$nvars ) {
    vv[[ii]]<-ncvar_get(nc,vnames[ii])
    if ( tolower(vnames[ii]) %in% c("time","date","time_bnds") ) {
      time.origin<-ncatt_get( nc, vnames[ii], attname="units")$value
      if ( time.origin == 0 ) time.origin<-ncatt_get( nc, "time", attname="units")$value
      #cat(vnames[ii],"; units=",time.origin,"\n")
      if ( tolower(vnames[ii]) == "time_bnds" & length(vv[[ii]]) > 2) {
       vv[[ii]]<-apply(vv[[ii]],c(2),mean)
      } else {
       vv[[ii]]<-mean(vv[[ii]])
      }
      if ( grepl("hours",time.origin) ) { 
        vv[[ii]]<-as.POSIXct(vv[[ii]]*3600,origin=gsub("hours since ","",time.origin))
      } else if ( grepl("days",time.origin) ) { 
        #print(gsub("days since ","",time.origin))
        vv[[ii]]<-as.POSIXct(vv[[ii]]*3600*24,origin=gsub("days since ","",time.origin))
      } else if ( grepl("seconds",time.origin) ) { 
        vv[[ii]]<-as.POSIXct(vv[[ii]],origin=gsub("seconds since ","",time.origin))
      } else if (time.origin == "YYYYMMDD UTC" ) {
        vv[[ii]]<-as.POSIXct(vv[[ii]]*3600,origin="1900-01-01 00:00:00.0")
      } else {
        cat(vnames[ii],time.origin,"not converted to date","\n")
      }
    }
  }
  names(vv)<-vnames
  nc_close(nc)

  op<-list(dims=dd,vars=vv,units=vunits)

}
