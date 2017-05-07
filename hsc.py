import numpy as np

def dict_normalization(D,P):
  dist=np.linalg.norm(D,axis=0);
  for i in range(P):
    D[:,i]=D[:,i]/dist[i];
  return D;

def dict_initializer(M,P):
  return dict_normalization(np.random.randn(M,P),P);

def sparse_coding(D,batch,FLAGS,group):
  A=np.zeros((FLAGS.P,FLAGS.batch_num),dtype=float);
  pre_mse=10;
  for i in range(FLAGS.sc_max_steps):
    A=A-(1.0/float(FLAGS.batch_num))*np.matmul(D.transpose(),(np.matmul(D,A)-batch));
    for j in range(len(A[0])):
      for k in range(len(group)):
        group_value=np.linalg.norm(A[group[k],j]);
        if(group_value>(1.0/float(FLAGS.batch_num))*FLAGS.sc_lambda*FLAGS.sc_Wg):
          A[group[k],j]=(group_value-(1.0/float(FLAGS.batch_num))*FLAGS.sc_lambda*FLAGS.sc_Wg)/group_value*A[group[k],j];
        else:
          A[group[k],j]=0.0;    
      # LASSO
      """
      for k in range(len(A)):
        if(abs(A[k,j])>(1.0/float(FLAGS.batch_num))*FLAGS.sc_lambda):
          A[k,j]=(abs(A[k,j])-(1.0/float(FLAGS.batch_num))*FLAGS.sc_lambda)*(abs(A[k,j])/A[k,j]);
        else:
          A[k,j]=0.0;
      """
    loss=np.linalg.norm(np.matmul(D,A)-batch,axis=0);mse=np.mean(loss);
    #print(str(i)+"th sparse coding MSE: "+str(mse));
    mse_diff=abs(mse-pre_mse);
    if(mse_diff<FLAGS.sc_mse_diff_threshold):
      break;
    pre_mse=mse;
  return A;

def dictionary_learning(D,batch,A,lr,FLAGS):
  D=D-lr*(1.0/float(FLAGS.batch_num))*np.matmul((np.matmul(D,A)-batch),A.transpose());
  return dict_normalization(D,FLAGS.P);  
