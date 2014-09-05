// provide supplementary functions

typedef double REAL;

REAL findMax(int n,REAL data[]){
  REAL locMax=data[0];
  int i;
  for(i=1;i<n;i++){
    if(data[i]>locMax)
      locMax=data[i];
  }
  return locMax;
} 
REAL findMin(int n,REAL data[]){
  REAL locMin=data[0];
  int i;
  for(i=1;i<n;i++){
    if(data[i]<locMin)
      locMin=data[i];
  }
  return locMin;
}
