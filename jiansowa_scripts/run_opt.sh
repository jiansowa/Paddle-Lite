# run in directory to save converted model

OPT="/home/jasonwang/base/jiansowa_github/Paddle-Lite/build.opt/lite/api/opt"
#MODELDIR='/home/jasonwang/models/paddle/MobileNetV1'

# print ops Paddle-Lite supported
#${OPT} --print_all_ops=true

# check platform compatibility of ops
#${OPT} --print_model_ops=true --model_dir=${MODELDIR} --valid_targets=x86

export GLOG_v=5

model='mobilenet_v1_post_quant'
#model='mobilenet_v1_fuse_post_quant'

if [ "${model}" == 'float' ]; then
  MODELDIR='/home/jasonwang/models/paddle/MobileNetV1'
  ${OPT} --model_dir=${MODELDIR}  \
  	--valid_targets=nna,x86 --optimize_out_type=naive_buffer --optimize_out=mobilenet_v1_opt
  ${OPT} --model_dir=${MODELDIR}  \
  	--valid_targets=nna,x86 --optimize_out_type=protobuf --optimize_out=mobilenet_v1_opt 

  #MODELDIR='/home/jasonwang/models/paddle/MobileNetV1_combined'
  #${OPT} --model_file=${MODELDIR}/__model__ --param_file=${MODELDIR}/params \
  #	--valid_targets=nna,x86 --optimize_out_type=naive_buffer --optimize_out=mobilenet_v1_opt 
  #${OPT} --model_file=${MODELDIR}/__model__ --param_file=${MODELDIR}/params \
  #  --valid_targets=nna,x86 --optimize_out_type=protobuf --optimize_out=mobilenet_v1_opt 
elif [ "${model}" == 'mobilenet_v1_post_quant' ]; then
  QuantizedModelDir='/home/jasonwang/models/paddle/MobileNetV1_post_quant'
  OptimizedOut='/home/jasonwang/models/paddle/paddle_lite_opt/mobilenet_v1_int8_opt_nna'
  ${OPT} --model_dir=${QuantizedModelDir} \
    --valid_targets=imagination_nna,arm --optimize_out_type=protobuf --optimize_out=${OptimizedOut}
  ${OPT} --model_dir=${QuantizedModelDir} \
    --valid_targets=imagination_nna,arm --optimize_out_type=naive_buffer --optimize_out=${OptimizedOut}
fi
