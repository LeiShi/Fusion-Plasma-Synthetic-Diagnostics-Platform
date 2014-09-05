

int loadNTProfiles(const char* fname,int** n_prof,REAL** a_p, REAL** n_p,REAL** Ti_p,REAL** Te_p){
  // netcdf identifiers
  int file_id; // id for file
  int a_dim_id;  // ids for dimensions
  int a_id,ne_id,Ti_id,Te_id; // ids for variables
  size_t *n;   
  int error_code;		// non-zero means netcdf error
     
  *n_prof=(int*)PyMem_Malloc(sizeof(int));
  n=(size_t *)PyMem_Malloc(sizeof(size_t));
  
  if ((error_code = nc_open(fname, NC_NOWRITE, &file_id))) // create & open the netcdf file
    ERR(error_code);
     
  if ((error_code = nc_inq_dimid(file_id, "profiles_radial_grid", &a_dim_id))) // get the dimension id
    ERR(error_code);
  if ((error_code = nc_inq_dimlen(file_id,a_dim_id,n)))
    ERR(error_code);

  if ((error_code = nc_inq_varid(file_id,"psi_grid_profiles",&a_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"ne_profile",&ne_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"Ti_profile",&Ti_id)))
    ERR(error_code);
  if ((error_code = nc_inq_varid(file_id,"Te_profile",&Te_id)))
    ERR(error_code);

  **n_prof = *n;
  *a_p=(REAL*)PyMem_Malloc(sizeof(REAL)* (*n));
  *n_p = (REAL*)PyMem_Malloc(sizeof(REAL)* (*n));
  *Ti_p = (REAL*)PyMem_Malloc(sizeof(REAL)* (*n));
  *Te_p = (REAL*)PyMem_Malloc(sizeof(REAL)* (*n));
  PyMem_Free(n);
  
  if ((error_code = nc_get_var_double(file_id,a_id,*a_p)))
    ERR(error_code);
  if ((error_code = nc_get_var_double(file_id,ne_id,*n_p)))
    ERR(error_code);
  if ((error_code = nc_get_var_double(file_id,Ti_id,*Ti_p)))
    ERR(error_code);
  if ((error_code = nc_get_var_double(file_id,Te_id,*Te_p)))
    ERR(error_code);

  return 0;
}
