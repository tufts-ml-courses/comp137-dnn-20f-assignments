��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.02v2.3.0-rc2-23-gb36436b0878��
�
embedding_11/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`
*(
shared_nameembedding_11/embeddings
�
+embedding_11/embeddings/Read/ReadVariableOpReadVariableOpembedding_11/embeddings*
_output_shapes

:`
*
dtype0
�
%simple_rnn_7/simple_rnn_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
_*6
shared_name'%simple_rnn_7/simple_rnn_cell_8/kernel
�
9simple_rnn_7/simple_rnn_cell_8/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_7/simple_rnn_cell_8/kernel*
_output_shapes

:
_*
dtype0
�
/simple_rnn_7/simple_rnn_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:__*@
shared_name1/simple_rnn_7/simple_rnn_cell_8/recurrent_kernel
�
Csimple_rnn_7/simple_rnn_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_7/simple_rnn_cell_8/recurrent_kernel*
_output_shapes

:__*
dtype0
�
#simple_rnn_7/simple_rnn_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:_*4
shared_name%#simple_rnn_7/simple_rnn_cell_8/bias
�
7simple_rnn_7/simple_rnn_cell_8/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_7/simple_rnn_cell_8/bias*
_output_shapes
:_*
dtype0
�
simple_rnn_7/VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape
:_*&
shared_namesimple_rnn_7/Variable

)simple_rnn_7/Variable/Read/ReadVariableOpReadVariableOpsimple_rnn_7/Variable*
_output_shapes

:_*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api

signatures
R
		variables

trainable_variables
regularization_losses
	keras_api
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api

0
1
2
3

0
1
2
3
 
�
layer_regularization_losses
	variables
layer_metrics
trainable_variables
non_trainable_variables

layers
metrics
regularization_losses
 
 
 
 
�
 layer_regularization_losses
		variables
!layer_metrics

trainable_variables

"layers
#metrics
regularization_losses
$non_trainable_variables
ge
VARIABLE_VALUEembedding_11/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�
%layer_regularization_losses
	variables
&layer_metrics
trainable_variables

'layers
(metrics
regularization_losses
)non_trainable_variables
~

kernel
recurrent_kernel
bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
 

0
1
2

0
1
2
 
�
.layer_regularization_losses
	variables
/layer_metrics

0states
trainable_variables
1non_trainable_variables

2layers
3metrics
regularization_losses
a_
VARIABLE_VALUE%simple_rnn_7/simple_rnn_cell_8/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/simple_rnn_7/simple_rnn_cell_8/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#simple_rnn_7/simple_rnn_cell_8/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
2
 
 
 
 
 
 
 
 
 
 
 

0
1
2

0
1
2
 
�
4layer_regularization_losses
*	variables
5layer_metrics
+trainable_variables

6layers
7metrics
,regularization_losses
8non_trainable_variables
 
 

90
 

0
 
 
 
 
 
 
mk
VARIABLE_VALUEsimple_rnn_7/VariableBlayer_with_weights-1/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUE
q
serving_default_input_12Placeholder*"
_output_shapes
:d*
dtype0*
shape:d
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_12embedding_11/embeddings%simple_rnn_7/simple_rnn_cell_8/kernel#simple_rnn_7/simple_rnn_cell_8/biassimple_rnn_7/Variable/simple_rnn_7/simple_rnn_cell_8/recurrent_kernel*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:d_*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_60272
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+embedding_11/embeddings/Read/ReadVariableOp9simple_rnn_7/simple_rnn_cell_8/kernel/Read/ReadVariableOpCsimple_rnn_7/simple_rnn_cell_8/recurrent_kernel/Read/ReadVariableOp7simple_rnn_7/simple_rnn_cell_8/bias/Read/ReadVariableOp)simple_rnn_7/Variable/Read/ReadVariableOpConst*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_61481
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_11/embeddings%simple_rnn_7/simple_rnn_cell_8/kernel/simple_rnn_7/simple_rnn_cell_8/recurrent_kernel#simple_rnn_7/simple_rnn_cell_8/biassimple_rnn_7/Variable*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_61506��
�

�
simple_rnn_7_while_cond_605686
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_26
2simple_rnn_7_while_less_simple_rnn_7_strided_sliceM
Isimple_rnn_7_while_simple_rnn_7_while_cond_60568___redundant_placeholder0M
Isimple_rnn_7_while_simple_rnn_7_while_cond_60568___redundant_placeholder1M
Isimple_rnn_7_while_simple_rnn_7_while_cond_60568___redundant_placeholder2M
Isimple_rnn_7_while_simple_rnn_7_while_cond_60568___redundant_placeholder3
simple_rnn_7_while_identity
�
simple_rnn_7/while/LessLesssimple_rnn_7_while_placeholder2simple_rnn_7_while_less_simple_rnn_7_strided_slice*
T0*
_output_shapes
: 2
simple_rnn_7/while/Less�
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_7/while/Identity"C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0*7
_input_shapes&
$: : : : :_: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
:
�	
�
1__inference_simple_rnn_cell_8_layer_call_fn_61443

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:_:_*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_594952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:_2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:_2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*3
_input_shapes"
 :
:_:::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:

 
_user_specified_nameinputs:HD

_output_shapes

:_
"
_user_specified_name
states/0
�
�
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_59478

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
_*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:_*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2	
BiasAdd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:__*
dtype02
MatMul_1/ReadVariableOpp
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:_2

MatMul_1b
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*
_output_shapes

:_2
addF
TanhTanhadd:z:0*
T0*
_output_shapes

:_2
TanhS
IdentityIdentityTanh:y:0*
T0*
_output_shapes

:_2

IdentityW

Identity_1IdentityTanh:y:0*
T0*
_output_shapes

:_2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*3
_input_shapes"
 :
:_::::F B

_output_shapes

:

 
_user_specified_nameinputs:FB

_output_shapes

:_
 
_user_specified_namestates
�
�
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_61315

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource&
"matmul_1_readvariableop_1_resource
identity

identity_1��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
_*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:_*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2	
BiasAddu
MatMul_1/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1/ReadVariableOp_1ReadVariableOp"matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02
MatMul_1/ReadVariableOp_1�
MatMul_1BatchMatMulV2MatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:2

MatMul_1[
addAddV2BiasAdd:output:0MatMul_1:output:0*
T0*
_output_shapes
:2
add@
TanhTanhadd:z:0*
T0*
_output_shapes
:2
TanhM
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:2

IdentityQ

Identity_1IdentityTanh:y:0*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*-
_input_shapes
:
:::::F B

_output_shapes

:

 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
�
�
-__inference_sequential_11_layer_call_fn_60509
input_12
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:d_*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_11_layer_call_and_return_conditional_losses_602102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������d:::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������d
"
_user_specified_name
input_12
�

�
simple_rnn_7_while_cond_603166
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_26
2simple_rnn_7_while_less_simple_rnn_7_strided_sliceM
Isimple_rnn_7_while_simple_rnn_7_while_cond_60316___redundant_placeholder0M
Isimple_rnn_7_while_simple_rnn_7_while_cond_60316___redundant_placeholder1M
Isimple_rnn_7_while_simple_rnn_7_while_cond_60316___redundant_placeholder2M
Isimple_rnn_7_while_simple_rnn_7_while_cond_60316___redundant_placeholder3
simple_rnn_7_while_identity
�
simple_rnn_7/while/LessLesssimple_rnn_7_while_placeholder2simple_rnn_7_while_less_simple_rnn_7_strided_slice*
T0*
_output_shapes
: 2
simple_rnn_7/while/Less�
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_7/while/Identity"C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0*7
_input_shapes&
$: : : : :_: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
:
�

�
simple_rnn_7_while_cond_604276
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_26
2simple_rnn_7_while_less_simple_rnn_7_strided_sliceM
Isimple_rnn_7_while_simple_rnn_7_while_cond_60427___redundant_placeholder0M
Isimple_rnn_7_while_simple_rnn_7_while_cond_60427___redundant_placeholder1M
Isimple_rnn_7_while_simple_rnn_7_while_cond_60427___redundant_placeholder2M
Isimple_rnn_7_while_simple_rnn_7_while_cond_60427___redundant_placeholder3
simple_rnn_7_while_identity
�
simple_rnn_7/while/LessLesssimple_rnn_7_while_placeholder2simple_rnn_7_while_less_simple_rnn_7_strided_slice*
T0*
_output_shapes
: 2
simple_rnn_7/while/Less�
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_7/while/Identity"C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0*7
_input_shapes&
$: : : : :_: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
:
�
�
,__inference_simple_rnn_7_layer_call_fn_61264

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:d_*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_600342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*1
_input_shapes 
:d
::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:d

 
_user_specified_nameinputs
�
r
,__inference_embedding_11_layer_call_fn_60817

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:d
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_11_layer_call_and_return_conditional_losses_599212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
:d
2

Identity"
identityIdentity:output:0*!
_input_shapes
:d:22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:d
 
_user_specified_nameinputs
�
`
D__inference_lambda_11_layer_call_and_return_conditional_losses_59891

inputs
identityS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
add/yX
addAddV2inputsadd/y:output:0*
T0*"
_output_shapes
:d2
addo
SqueezeSqueezeadd:z:0*
T0*
_output_shapes

:d*
squeeze_dims

���������2	
Squeeze[
IdentityIdentitySqueeze:output:0*
T0*
_output_shapes

:d2

Identity"
identityIdentity:output:0*!
_input_shapes
:d:J F
"
_output_shapes
:d
 
_user_specified_nameinputs
�	
�
1__inference_simple_rnn_cell_8_layer_call_fn_61429

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:_:_*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_594782
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:_2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes

:_2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*3
_input_shapes"
 :
:_:::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:

 
_user_specified_nameinputs:HD

_output_shapes

:_
"
_user_specified_name
states/0
�
�
while_cond_59696
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_59696___redundant_placeholder03
/while_while_cond_59696___redundant_placeholder13
/while_while_cond_59696___redundant_placeholder23
/while_while_cond_59696___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*7
_input_shapes&
$: : : : :_: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_60954
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_60954___redundant_placeholder03
/while_while_cond_60954___redundant_placeholder13
/while_while_cond_60954___redundant_placeholder23
/while_while_cond_60954___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*7
_input_shapes&
$: : : : :_: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
:
�<
�
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_60034

inputs4
0simple_rnn_cell_8_matmul_readvariableop_resource5
1simple_rnn_cell_8_biasadd_readvariableop_resource6
2simple_rnn_cell_8_matmul_1_readvariableop_resource8
4simple_rnn_cell_8_matmul_1_readvariableop_1_resource
identity��AssignVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permq
	transpose	Transposeinputstranspose/perm:output:0*
T0*"
_output_shapes
:d
2
	transposec
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d      
   2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
*
shrink_axis_mask2
strided_slice_1�
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

:
_*
dtype02)
'simple_rnn_cell_8/MatMul/ReadVariableOp�
simple_rnn_cell_8/MatMulMatMulstrided_slice_1:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/MatMul�
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype02*
(simple_rnn_cell_8/BiasAdd/ReadVariableOp�
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/BiasAdd�
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02+
)simple_rnn_cell_8/MatMul_1/ReadVariableOp�
+simple_rnn_cell_8/MatMul_1/ReadVariableOp_1ReadVariableOp4simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02-
+simple_rnn_cell_8/MatMul_1/ReadVariableOp_1�
simple_rnn_cell_8/MatMul_1MatMul1simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:03simple_rnn_cell_8/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/MatMul_1�
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/add|
simple_rnn_cell_8/TanhTanhsimple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time�
ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02
ReadVariableOp
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource4simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :_: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_59968*
condR
while_cond_59967*/
output_shapes
: : : : :_: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:d_*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:_*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*"
_output_shapes
:d_2
transpose_1�
AssignVariableOpAssignVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resourcewhile:output:4^ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpy
IdentityIdentitytranspose_1:y:0^AssignVariableOp^while*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*1
_input_shapes 
:d
::::2$
AssignVariableOpAssignVariableOp2
whilewhile:J F
"
_output_shapes
:d

 
_user_specified_nameinputs
�
�
__inference__traced_save_61481
file_prefix6
2savev2_embedding_11_embeddings_read_readvariableopD
@savev2_simple_rnn_7_simple_rnn_cell_8_kernel_read_readvariableopN
Jsavev2_simple_rnn_7_simple_rnn_cell_8_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_7_simple_rnn_cell_8_bias_read_readvariableop4
0savev2_simple_rnn_7_variable_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_c118af3f7b09459e998f6f0608ec7999/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-1/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_embedding_11_embeddings_read_readvariableop@savev2_simple_rnn_7_simple_rnn_cell_8_kernel_read_readvariableopJsavev2_simple_rnn_7_simple_rnn_cell_8_recurrent_kernel_read_readvariableop>savev2_simple_rnn_7_simple_rnn_cell_8_bias_read_readvariableop0savev2_simple_rnn_7_variable_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*E
_input_shapes4
2: :`
:
_:__:_:_: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:`
:$ 

_output_shapes

:
_:$ 

_output_shapes

:__: 

_output_shapes
:_:$ 

_output_shapes

:_:

_output_shapes
: 
�U
�
H__inference_sequential_11_layer_call_and_return_conditional_losses_60746

inputs'
#embedding_11_embedding_lookup_60642A
=simple_rnn_7_simple_rnn_cell_8_matmul_readvariableop_resourceB
>simple_rnn_7_simple_rnn_cell_8_biasadd_readvariableop_resourceC
?simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resourceE
Asimple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_1_resource
identity��simple_rnn_7/AssignVariableOp�simple_rnn_7/whileg
lambda_11/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
lambda_11/add/yv
lambda_11/addAddV2inputslambda_11/add/y:output:0*
T0*"
_output_shapes
:d2
lambda_11/add�
lambda_11/SqueezeSqueezelambda_11/add:z:0*
T0*
_output_shapes

:d*
squeeze_dims

���������2
lambda_11/Squeeze�
embedding_11/CastCastlambda_11/Squeeze:output:0*

DstT0*

SrcT0*
_output_shapes

:d2
embedding_11/Cast�
embedding_11/embedding_lookupResourceGather#embedding_11_embedding_lookup_60642embedding_11/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_11/embedding_lookup/60642*"
_output_shapes
:d
*
dtype02
embedding_11/embedding_lookup�
&embedding_11/embedding_lookup/IdentityIdentity&embedding_11/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_11/embedding_lookup/60642*"
_output_shapes
:d
2(
&embedding_11/embedding_lookup/Identity�
(embedding_11/embedding_lookup/Identity_1Identity/embedding_11/embedding_lookup/Identity:output:0*
T0*"
_output_shapes
:d
2*
(embedding_11/embedding_lookup/Identity_1�
simple_rnn_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_7/transpose/perm�
simple_rnn_7/transpose	Transpose1embedding_11/embedding_lookup/Identity_1:output:0$simple_rnn_7/transpose/perm:output:0*
T0*"
_output_shapes
:d
2
simple_rnn_7/transpose}
simple_rnn_7/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d      
   2
simple_rnn_7/Shape�
 simple_rnn_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_7/strided_slice/stack�
"simple_rnn_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_7/strided_slice/stack_1�
"simple_rnn_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_7/strided_slice/stack_2�
simple_rnn_7/strided_sliceStridedSlicesimple_rnn_7/Shape:output:0)simple_rnn_7/strided_slice/stack:output:0+simple_rnn_7/strided_slice/stack_1:output:0+simple_rnn_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_7/strided_slice�
(simple_rnn_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(simple_rnn_7/TensorArrayV2/element_shape�
simple_rnn_7/TensorArrayV2TensorListReserve1simple_rnn_7/TensorArrayV2/element_shape:output:0#simple_rnn_7/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_7/TensorArrayV2�
Bsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   2D
Bsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape�
4simple_rnn_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_7/transpose:y:0Ksimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_7/TensorArrayUnstack/TensorListFromTensor�
"simple_rnn_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_7/strided_slice_1/stack�
$simple_rnn_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_7/strided_slice_1/stack_1�
$simple_rnn_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_7/strided_slice_1/stack_2�
simple_rnn_7/strided_slice_1StridedSlicesimple_rnn_7/transpose:y:0+simple_rnn_7/strided_slice_1/stack:output:0-simple_rnn_7/strided_slice_1/stack_1:output:0-simple_rnn_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
*
shrink_axis_mask2
simple_rnn_7/strided_slice_1�
4simple_rnn_7/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp=simple_rnn_7_simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

:
_*
dtype026
4simple_rnn_7/simple_rnn_cell_8/MatMul/ReadVariableOp�
%simple_rnn_7/simple_rnn_cell_8/MatMulMatMul%simple_rnn_7/strided_slice_1:output:0<simple_rnn_7/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2'
%simple_rnn_7/simple_rnn_cell_8/MatMul�
5simple_rnn_7/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_7_simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype027
5simple_rnn_7/simple_rnn_cell_8/BiasAdd/ReadVariableOp�
&simple_rnn_7/simple_rnn_cell_8/BiasAddBiasAdd/simple_rnn_7/simple_rnn_cell_8/MatMul:product:0=simple_rnn_7/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2(
&simple_rnn_7/simple_rnn_cell_8/BiasAdd�
6simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype028
6simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp�
8simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp_1ReadVariableOpAsimple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02:
8simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp_1�
'simple_rnn_7/simple_rnn_cell_8/MatMul_1MatMul>simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0@simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:_2)
'simple_rnn_7/simple_rnn_cell_8/MatMul_1�
"simple_rnn_7/simple_rnn_cell_8/addAddV2/simple_rnn_7/simple_rnn_cell_8/BiasAdd:output:01simple_rnn_7/simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2$
"simple_rnn_7/simple_rnn_cell_8/add�
#simple_rnn_7/simple_rnn_cell_8/TanhTanh&simple_rnn_7/simple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2%
#simple_rnn_7/simple_rnn_cell_8/Tanh�
*simple_rnn_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2,
*simple_rnn_7/TensorArrayV2_1/element_shape�
simple_rnn_7/TensorArrayV2_1TensorListReserve3simple_rnn_7/TensorArrayV2_1/element_shape:output:0#simple_rnn_7/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_7/TensorArrayV2_1h
simple_rnn_7/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_7/time�
simple_rnn_7/ReadVariableOpReadVariableOp?simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02
simple_rnn_7/ReadVariableOp�
%simple_rnn_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%simple_rnn_7/while/maximum_iterations�
simple_rnn_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_7/while/loop_counter�
simple_rnn_7/whileWhile(simple_rnn_7/while/loop_counter:output:0.simple_rnn_7/while/maximum_iterations:output:0simple_rnn_7/time:output:0%simple_rnn_7/TensorArrayV2_1:handle:0#simple_rnn_7/ReadVariableOp:value:0#simple_rnn_7/strided_slice:output:0Dsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_7_simple_rnn_cell_8_matmul_readvariableop_resource>simple_rnn_7_simple_rnn_cell_8_biasadd_readvariableop_resourceAsimple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :_: : : : : *%
_read_only_resource_inputs
	*)
body!R
simple_rnn_7_while_body_60680*)
cond!R
simple_rnn_7_while_cond_60679*/
output_shapes
: : : : :_: : : : : *
parallel_iterations 2
simple_rnn_7/while�
=simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2?
=simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape�
/simple_rnn_7/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_7/while:output:3Fsimple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:d_*
element_dtype021
/simple_rnn_7/TensorArrayV2Stack/TensorListStack�
"simple_rnn_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2$
"simple_rnn_7/strided_slice_2/stack�
$simple_rnn_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_7/strided_slice_2/stack_1�
$simple_rnn_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_7/strided_slice_2/stack_2�
simple_rnn_7/strided_slice_2StridedSlice8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_7/strided_slice_2/stack:output:0-simple_rnn_7/strided_slice_2/stack_1:output:0-simple_rnn_7/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:_*
shrink_axis_mask2
simple_rnn_7/strided_slice_2�
simple_rnn_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_7/transpose_1/perm�
simple_rnn_7/transpose_1	Transpose8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_7/transpose_1/perm:output:0*
T0*"
_output_shapes
:d_2
simple_rnn_7/transpose_1�
simple_rnn_7/AssignVariableOpAssignVariableOp?simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resourcesimple_rnn_7/while:output:4^simple_rnn_7/ReadVariableOp7^simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
_output_shapes
 *
dtype02
simple_rnn_7/AssignVariableOp�
IdentityIdentitysimple_rnn_7/transpose_1:y:0^simple_rnn_7/AssignVariableOp^simple_rnn_7/while*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*5
_input_shapes$
":d:::::2>
simple_rnn_7/AssignVariableOpsimple_rnn_7/AssignVariableOp2(
simple_rnn_7/whilesimple_rnn_7/while:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�)
�
while_body_61185
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_8_matmul_readvariableop_resource;
7while_simple_rnn_cell_8_biasadd_readvariableop_resource<
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource��
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:
_*
dtype02/
-while/simple_rnn_cell_8/MatMul/ReadVariableOp�
while/simple_rnn_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2 
while/simple_rnn_cell_8/MatMul�
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:_*
dtype020
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2!
while/simple_rnn_cell_8/BiasAdd�
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:__*
dtype021
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_8/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:_2"
 while/simple_rnn_cell_8/MatMul_1�
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2
while/simple_rnn_cell_8/add�
while/simple_rnn_cell_8/TanhTanhwhile/simple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2
while/simple_rnn_cell_8/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_8/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3{
while/Identity_4Identity while/simple_rnn_cell_8/Tanh:y:0*
T0*
_output_shapes

:_2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*5
_input_shapes$
": : : : :_: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
: 
�
�
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_59393

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource&
"matmul_1_readvariableop_1_resource
identity

identity_1��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
_*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:_*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2	
BiasAddy
MatMul_1/ReadVariableOpReadVariableOpstates*
_output_shapes

:_*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1/ReadVariableOp_1ReadVariableOp"matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02
MatMul_1/ReadVariableOp_1�
MatMul_1MatMulMatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:_2

MatMul_1b
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*
_output_shapes

:_2
addF
TanhTanhadd:z:0*
T0*
_output_shapes

:_2
TanhS
IdentityIdentityTanh:y:0*
T0*
_output_shapes

:_2

IdentityW

Identity_1IdentityTanh:y:0*
T0*
_output_shapes

:_2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*-
_input_shapes
:
:::::F B

_output_shapes

:

 
_user_specified_nameinputs:&"
 
_user_specified_namestates
�<
�
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_60919
inputs_04
0simple_rnn_cell_8_matmul_readvariableop_resource5
1simple_rnn_cell_8_biasadd_readvariableop_resource6
2simple_rnn_cell_8_matmul_1_readvariableop_resource8
4simple_rnn_cell_8_matmul_1_readvariableop_1_resource
identity��AssignVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*+
_output_shapes
:���������
2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
*
shrink_axis_mask2
strided_slice_1�
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

:
_*
dtype02)
'simple_rnn_cell_8/MatMul/ReadVariableOp�
simple_rnn_cell_8/MatMulMatMulstrided_slice_1:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/MatMul�
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype02*
(simple_rnn_cell_8/BiasAdd/ReadVariableOp�
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/BiasAdd�
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02+
)simple_rnn_cell_8/MatMul_1/ReadVariableOp�
+simple_rnn_cell_8/MatMul_1/ReadVariableOp_1ReadVariableOp4simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02-
+simple_rnn_cell_8/MatMul_1/ReadVariableOp_1�
simple_rnn_cell_8/MatMul_1MatMul1simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:03simple_rnn_cell_8/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/MatMul_1�
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/add|
simple_rnn_cell_8/TanhTanhsimple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time�
ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02
ReadVariableOp
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource4simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :_: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_60853*
condR
while_cond_60852*/
output_shapes
: : : : :_: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������_*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:_*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������_2
transpose_1�
AssignVariableOpAssignVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resourcewhile:output:4^ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp�
IdentityIdentitytranspose_1:y:0^AssignVariableOp^while*
T0*+
_output_shapes
:���������_2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������
::::2$
AssignVariableOpAssignVariableOp2
whilewhile:U Q
+
_output_shapes
:���������

"
_user_specified_name
inputs/0
�6
�	
simple_rnn_7_while_body_603176
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_23
/simple_rnn_7_while_simple_rnn_7_strided_slice_0q
msimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resource_0J
Fsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0K
Gsimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
simple_rnn_7_while_identity!
simple_rnn_7_while_identity_1!
simple_rnn_7_while_identity_2!
simple_rnn_7_while_identity_3!
simple_rnn_7_while_identity_41
-simple_rnn_7_while_simple_rnn_7_strided_sliceo
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resourceH
Dsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resourceI
Esimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource��
Dsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   2F
Dsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape�
6simple_rnn_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_7_while_placeholderMsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
*
element_dtype028
6simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem�
:simple_rnn_7/while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:
_*
dtype02<
:simple_rnn_7/while/simple_rnn_cell_8/MatMul/ReadVariableOp�
+simple_rnn_7/while/simple_rnn_cell_8/MatMulMatMul=simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_7/while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2-
+simple_rnn_7/while/simple_rnn_cell_8/MatMul�
;simple_rnn_7/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:_*
dtype02=
;simple_rnn_7/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp�
,simple_rnn_7/while/simple_rnn_cell_8/BiasAddBiasAdd5simple_rnn_7/while/simple_rnn_cell_8/MatMul:product:0Csimple_rnn_7/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2.
,simple_rnn_7/while/simple_rnn_cell_8/BiasAdd�
<simple_rnn_7/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:__*
dtype02>
<simple_rnn_7/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp�
-simple_rnn_7/while/simple_rnn_cell_8/MatMul_1MatMul simple_rnn_7_while_placeholder_2Dsimple_rnn_7/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:_2/
-simple_rnn_7/while/simple_rnn_cell_8/MatMul_1�
(simple_rnn_7/while/simple_rnn_cell_8/addAddV25simple_rnn_7/while/simple_rnn_cell_8/BiasAdd:output:07simple_rnn_7/while/simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2*
(simple_rnn_7/while/simple_rnn_cell_8/add�
)simple_rnn_7/while/simple_rnn_cell_8/TanhTanh,simple_rnn_7/while/simple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2+
)simple_rnn_7/while/simple_rnn_cell_8/Tanh�
7simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_7_while_placeholder_1simple_rnn_7_while_placeholder-simple_rnn_7/while/simple_rnn_cell_8/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_7/while/add/y�
simple_rnn_7/while/addAddV2simple_rnn_7_while_placeholder!simple_rnn_7/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/addz
simple_rnn_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_7/while/add_1/y�
simple_rnn_7/while/add_1AddV22simple_rnn_7_while_simple_rnn_7_while_loop_counter#simple_rnn_7/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/add_1�
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/add_1:z:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/Identity�
simple_rnn_7/while/Identity_1Identity8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations*
T0*
_output_shapes
: 2
simple_rnn_7/while/Identity_1�
simple_rnn_7/while/Identity_2Identitysimple_rnn_7/while/add:z:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/Identity_2�
simple_rnn_7/while/Identity_3IdentityGsimple_rnn_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/Identity_3�
simple_rnn_7/while/Identity_4Identity-simple_rnn_7/while/simple_rnn_cell_8/Tanh:y:0*
T0*
_output_shapes

:_2
simple_rnn_7/while/Identity_4"C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0"G
simple_rnn_7_while_identity_1&simple_rnn_7/while/Identity_1:output:0"G
simple_rnn_7_while_identity_2&simple_rnn_7/while/Identity_2:output:0"G
simple_rnn_7_while_identity_3&simple_rnn_7/while/Identity_3:output:0"G
simple_rnn_7_while_identity_4&simple_rnn_7/while/Identity_4:output:0"`
-simple_rnn_7_while_simple_rnn_7_strided_slice/simple_rnn_7_while_simple_rnn_7_strided_slice_0"�
Dsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resourceFsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"�
Esimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resourceGsimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"�
Csimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resourceEsimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resource_0"�
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensormsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0*5
_input_shapes$
": : : : :_: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
: 
�
�
1__inference_simple_rnn_cell_8_layer_call_fn_61348

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_613372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*-
_input_shapes
:
::::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:

 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
�C
�
+sequential_11_simple_rnn_7_while_body_59272R
Nsequential_11_simple_rnn_7_while_sequential_11_simple_rnn_7_while_loop_counterX
Tsequential_11_simple_rnn_7_while_sequential_11_simple_rnn_7_while_maximum_iterations0
,sequential_11_simple_rnn_7_while_placeholder2
.sequential_11_simple_rnn_7_while_placeholder_12
.sequential_11_simple_rnn_7_while_placeholder_2O
Ksequential_11_simple_rnn_7_while_sequential_11_simple_rnn_7_strided_slice_0�
�sequential_11_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_11_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0W
Ssequential_11_simple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resource_0X
Tsequential_11_simple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0Y
Usequential_11_simple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0-
)sequential_11_simple_rnn_7_while_identity/
+sequential_11_simple_rnn_7_while_identity_1/
+sequential_11_simple_rnn_7_while_identity_2/
+sequential_11_simple_rnn_7_while_identity_3/
+sequential_11_simple_rnn_7_while_identity_4M
Isequential_11_simple_rnn_7_while_sequential_11_simple_rnn_7_strided_slice�
�sequential_11_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_11_simple_rnn_7_tensorarrayunstack_tensorlistfromtensorU
Qsequential_11_simple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resourceV
Rsequential_11_simple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resourceW
Ssequential_11_simple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource��
Rsequential_11/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   2T
Rsequential_11/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape�
Dsequential_11/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_11_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_11_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0,sequential_11_simple_rnn_7_while_placeholder[sequential_11/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
*
element_dtype02F
Dsequential_11/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem�
Hsequential_11/simple_rnn_7/while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOpSsequential_11_simple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:
_*
dtype02J
Hsequential_11/simple_rnn_7/while/simple_rnn_cell_8/MatMul/ReadVariableOp�
9sequential_11/simple_rnn_7/while/simple_rnn_cell_8/MatMulMatMulKsequential_11/simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:0Psequential_11/simple_rnn_7/while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2;
9sequential_11/simple_rnn_7/while/simple_rnn_cell_8/MatMul�
Isequential_11/simple_rnn_7/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOpTsequential_11_simple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:_*
dtype02K
Isequential_11/simple_rnn_7/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp�
:sequential_11/simple_rnn_7/while/simple_rnn_cell_8/BiasAddBiasAddCsequential_11/simple_rnn_7/while/simple_rnn_cell_8/MatMul:product:0Qsequential_11/simple_rnn_7/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2<
:sequential_11/simple_rnn_7/while/simple_rnn_cell_8/BiasAdd�
Jsequential_11/simple_rnn_7/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOpUsequential_11_simple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:__*
dtype02L
Jsequential_11/simple_rnn_7/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp�
;sequential_11/simple_rnn_7/while/simple_rnn_cell_8/MatMul_1MatMul.sequential_11_simple_rnn_7_while_placeholder_2Rsequential_11/simple_rnn_7/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:_2=
;sequential_11/simple_rnn_7/while/simple_rnn_cell_8/MatMul_1�
6sequential_11/simple_rnn_7/while/simple_rnn_cell_8/addAddV2Csequential_11/simple_rnn_7/while/simple_rnn_cell_8/BiasAdd:output:0Esequential_11/simple_rnn_7/while/simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_28
6sequential_11/simple_rnn_7/while/simple_rnn_cell_8/add�
7sequential_11/simple_rnn_7/while/simple_rnn_cell_8/TanhTanh:sequential_11/simple_rnn_7/while/simple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_29
7sequential_11/simple_rnn_7/while/simple_rnn_cell_8/Tanh�
Esequential_11/simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_11_simple_rnn_7_while_placeholder_1,sequential_11_simple_rnn_7_while_placeholder;sequential_11/simple_rnn_7/while/simple_rnn_cell_8/Tanh:y:0*
_output_shapes
: *
element_dtype02G
Esequential_11/simple_rnn_7/while/TensorArrayV2Write/TensorListSetItem�
&sequential_11/simple_rnn_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_11/simple_rnn_7/while/add/y�
$sequential_11/simple_rnn_7/while/addAddV2,sequential_11_simple_rnn_7_while_placeholder/sequential_11/simple_rnn_7/while/add/y:output:0*
T0*
_output_shapes
: 2&
$sequential_11/simple_rnn_7/while/add�
(sequential_11/simple_rnn_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_11/simple_rnn_7/while/add_1/y�
&sequential_11/simple_rnn_7/while/add_1AddV2Nsequential_11_simple_rnn_7_while_sequential_11_simple_rnn_7_while_loop_counter1sequential_11/simple_rnn_7/while/add_1/y:output:0*
T0*
_output_shapes
: 2(
&sequential_11/simple_rnn_7/while/add_1�
)sequential_11/simple_rnn_7/while/IdentityIdentity*sequential_11/simple_rnn_7/while/add_1:z:0*
T0*
_output_shapes
: 2+
)sequential_11/simple_rnn_7/while/Identity�
+sequential_11/simple_rnn_7/while/Identity_1IdentityTsequential_11_simple_rnn_7_while_sequential_11_simple_rnn_7_while_maximum_iterations*
T0*
_output_shapes
: 2-
+sequential_11/simple_rnn_7/while/Identity_1�
+sequential_11/simple_rnn_7/while/Identity_2Identity(sequential_11/simple_rnn_7/while/add:z:0*
T0*
_output_shapes
: 2-
+sequential_11/simple_rnn_7/while/Identity_2�
+sequential_11/simple_rnn_7/while/Identity_3IdentityUsequential_11/simple_rnn_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2-
+sequential_11/simple_rnn_7/while/Identity_3�
+sequential_11/simple_rnn_7/while/Identity_4Identity;sequential_11/simple_rnn_7/while/simple_rnn_cell_8/Tanh:y:0*
T0*
_output_shapes

:_2-
+sequential_11/simple_rnn_7/while/Identity_4"_
)sequential_11_simple_rnn_7_while_identity2sequential_11/simple_rnn_7/while/Identity:output:0"c
+sequential_11_simple_rnn_7_while_identity_14sequential_11/simple_rnn_7/while/Identity_1:output:0"c
+sequential_11_simple_rnn_7_while_identity_24sequential_11/simple_rnn_7/while/Identity_2:output:0"c
+sequential_11_simple_rnn_7_while_identity_34sequential_11/simple_rnn_7/while/Identity_3:output:0"c
+sequential_11_simple_rnn_7_while_identity_44sequential_11/simple_rnn_7/while/Identity_4:output:0"�
Isequential_11_simple_rnn_7_while_sequential_11_simple_rnn_7_strided_sliceKsequential_11_simple_rnn_7_while_sequential_11_simple_rnn_7_strided_slice_0"�
Rsequential_11_simple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resourceTsequential_11_simple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"�
Ssequential_11_simple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resourceUsequential_11_simple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"�
Qsequential_11_simple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resourceSsequential_11_simple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resource_0"�
�sequential_11_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_11_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor�sequential_11_simple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_sequential_11_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0*5
_input_shapes$
": : : : :_: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
: 
�"
�
while_body_59697
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_8_59719_0#
while_simple_rnn_cell_8_59721_0#
while_simple_rnn_cell_8_59723_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_8_59719!
while_simple_rnn_cell_8_59721!
while_simple_rnn_cell_8_59723��/while/simple_rnn_cell_8/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
/while/simple_rnn_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_8_59719_0while_simple_rnn_cell_8_59721_0while_simple_rnn_cell_8_59723_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:_:_*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_5947821
/while/simple_rnn_cell_8/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_8/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity8while/simple_rnn_cell_8/StatefulPartitionedCall:output:10^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes

:_2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_8_59719while_simple_rnn_cell_8_59719_0"@
while_simple_rnn_cell_8_59721while_simple_rnn_cell_8_59721_0"@
while_simple_rnn_cell_8_59723while_simple_rnn_cell_8_59723_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*5
_input_shapes$
": : : : :_: : :::2b
/while/simple_rnn_cell_8/StatefulPartitionedCall/while/simple_rnn_cell_8/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
: 
�

�
simple_rnn_7_while_cond_606796
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_26
2simple_rnn_7_while_less_simple_rnn_7_strided_sliceM
Isimple_rnn_7_while_simple_rnn_7_while_cond_60679___redundant_placeholder0M
Isimple_rnn_7_while_simple_rnn_7_while_cond_60679___redundant_placeholder1M
Isimple_rnn_7_while_simple_rnn_7_while_cond_60679___redundant_placeholder2M
Isimple_rnn_7_while_simple_rnn_7_while_cond_60679___redundant_placeholder3
simple_rnn_7_while_identity
�
simple_rnn_7/while/LessLesssimple_rnn_7_while_placeholder2simple_rnn_7_while_less_simple_rnn_7_strided_slice*
T0*
_output_shapes
: 2
simple_rnn_7/while/Less�
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/Less:z:0*
T0
*
_output_shapes
: 2
simple_rnn_7/while/Identity"C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0*7
_input_shapes&
$: : : : :_: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
:
�6
�	
simple_rnn_7_while_body_605696
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_23
/simple_rnn_7_while_simple_rnn_7_strided_slice_0q
msimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resource_0J
Fsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0K
Gsimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
simple_rnn_7_while_identity!
simple_rnn_7_while_identity_1!
simple_rnn_7_while_identity_2!
simple_rnn_7_while_identity_3!
simple_rnn_7_while_identity_41
-simple_rnn_7_while_simple_rnn_7_strided_sliceo
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resourceH
Dsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resourceI
Esimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource��
Dsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   2F
Dsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape�
6simple_rnn_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_7_while_placeholderMsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
*
element_dtype028
6simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem�
:simple_rnn_7/while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:
_*
dtype02<
:simple_rnn_7/while/simple_rnn_cell_8/MatMul/ReadVariableOp�
+simple_rnn_7/while/simple_rnn_cell_8/MatMulMatMul=simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_7/while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2-
+simple_rnn_7/while/simple_rnn_cell_8/MatMul�
;simple_rnn_7/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:_*
dtype02=
;simple_rnn_7/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp�
,simple_rnn_7/while/simple_rnn_cell_8/BiasAddBiasAdd5simple_rnn_7/while/simple_rnn_cell_8/MatMul:product:0Csimple_rnn_7/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2.
,simple_rnn_7/while/simple_rnn_cell_8/BiasAdd�
<simple_rnn_7/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:__*
dtype02>
<simple_rnn_7/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp�
-simple_rnn_7/while/simple_rnn_cell_8/MatMul_1MatMul simple_rnn_7_while_placeholder_2Dsimple_rnn_7/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:_2/
-simple_rnn_7/while/simple_rnn_cell_8/MatMul_1�
(simple_rnn_7/while/simple_rnn_cell_8/addAddV25simple_rnn_7/while/simple_rnn_cell_8/BiasAdd:output:07simple_rnn_7/while/simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2*
(simple_rnn_7/while/simple_rnn_cell_8/add�
)simple_rnn_7/while/simple_rnn_cell_8/TanhTanh,simple_rnn_7/while/simple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2+
)simple_rnn_7/while/simple_rnn_cell_8/Tanh�
7simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_7_while_placeholder_1simple_rnn_7_while_placeholder-simple_rnn_7/while/simple_rnn_cell_8/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_7/while/add/y�
simple_rnn_7/while/addAddV2simple_rnn_7_while_placeholder!simple_rnn_7/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/addz
simple_rnn_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_7/while/add_1/y�
simple_rnn_7/while/add_1AddV22simple_rnn_7_while_simple_rnn_7_while_loop_counter#simple_rnn_7/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/add_1�
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/add_1:z:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/Identity�
simple_rnn_7/while/Identity_1Identity8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations*
T0*
_output_shapes
: 2
simple_rnn_7/while/Identity_1�
simple_rnn_7/while/Identity_2Identitysimple_rnn_7/while/add:z:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/Identity_2�
simple_rnn_7/while/Identity_3IdentityGsimple_rnn_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/Identity_3�
simple_rnn_7/while/Identity_4Identity-simple_rnn_7/while/simple_rnn_cell_8/Tanh:y:0*
T0*
_output_shapes

:_2
simple_rnn_7/while/Identity_4"C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0"G
simple_rnn_7_while_identity_1&simple_rnn_7/while/Identity_1:output:0"G
simple_rnn_7_while_identity_2&simple_rnn_7/while/Identity_2:output:0"G
simple_rnn_7_while_identity_3&simple_rnn_7/while/Identity_3:output:0"G
simple_rnn_7_while_identity_4&simple_rnn_7/while/Identity_4:output:0"`
-simple_rnn_7_while_simple_rnn_7_strided_slice/simple_rnn_7_while_simple_rnn_7_strided_slice_0"�
Dsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resourceFsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"�
Esimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resourceGsimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"�
Csimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resourceEsimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resource_0"�
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensormsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0*5
_input_shapes$
": : : : :_: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
: 
�
�
#__inference_signature_wrapper_60272
input_12
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:d_*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_593382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*5
_input_shapes$
":d:::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
"
_output_shapes
:d
"
_user_specified_name
input_12
�
�
-__inference_sequential_11_layer_call_fn_60524
input_12
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_12unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:d_*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_11_layer_call_and_return_conditional_losses_602422
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������d:::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������d
"
_user_specified_name
input_12
�
`
D__inference_lambda_11_layer_call_and_return_conditional_losses_60790

inputs
identityS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
add/yX
addAddV2inputsadd/y:output:0*
T0*"
_output_shapes
:d2
addo
SqueezeSqueezeadd:z:0*
T0*
_output_shapes

:d*
squeeze_dims

���������2	
Squeeze[
IdentityIdentitySqueeze:output:0*
T0*
_output_shapes

:d2

Identity"
identityIdentity:output:0*!
_input_shapes
:d:J F
"
_output_shapes
:d
 
_user_specified_nameinputs
�6
�	
simple_rnn_7_while_body_604286
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_23
/simple_rnn_7_while_simple_rnn_7_strided_slice_0q
msimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resource_0J
Fsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0K
Gsimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
simple_rnn_7_while_identity!
simple_rnn_7_while_identity_1!
simple_rnn_7_while_identity_2!
simple_rnn_7_while_identity_3!
simple_rnn_7_while_identity_41
-simple_rnn_7_while_simple_rnn_7_strided_sliceo
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resourceH
Dsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resourceI
Esimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource��
Dsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   2F
Dsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape�
6simple_rnn_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_7_while_placeholderMsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
*
element_dtype028
6simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem�
:simple_rnn_7/while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:
_*
dtype02<
:simple_rnn_7/while/simple_rnn_cell_8/MatMul/ReadVariableOp�
+simple_rnn_7/while/simple_rnn_cell_8/MatMulMatMul=simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_7/while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2-
+simple_rnn_7/while/simple_rnn_cell_8/MatMul�
;simple_rnn_7/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:_*
dtype02=
;simple_rnn_7/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp�
,simple_rnn_7/while/simple_rnn_cell_8/BiasAddBiasAdd5simple_rnn_7/while/simple_rnn_cell_8/MatMul:product:0Csimple_rnn_7/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2.
,simple_rnn_7/while/simple_rnn_cell_8/BiasAdd�
<simple_rnn_7/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:__*
dtype02>
<simple_rnn_7/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp�
-simple_rnn_7/while/simple_rnn_cell_8/MatMul_1MatMul simple_rnn_7_while_placeholder_2Dsimple_rnn_7/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:_2/
-simple_rnn_7/while/simple_rnn_cell_8/MatMul_1�
(simple_rnn_7/while/simple_rnn_cell_8/addAddV25simple_rnn_7/while/simple_rnn_cell_8/BiasAdd:output:07simple_rnn_7/while/simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2*
(simple_rnn_7/while/simple_rnn_cell_8/add�
)simple_rnn_7/while/simple_rnn_cell_8/TanhTanh,simple_rnn_7/while/simple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2+
)simple_rnn_7/while/simple_rnn_cell_8/Tanh�
7simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_7_while_placeholder_1simple_rnn_7_while_placeholder-simple_rnn_7/while/simple_rnn_cell_8/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_7/while/add/y�
simple_rnn_7/while/addAddV2simple_rnn_7_while_placeholder!simple_rnn_7/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/addz
simple_rnn_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_7/while/add_1/y�
simple_rnn_7/while/add_1AddV22simple_rnn_7_while_simple_rnn_7_while_loop_counter#simple_rnn_7/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/add_1�
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/add_1:z:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/Identity�
simple_rnn_7/while/Identity_1Identity8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations*
T0*
_output_shapes
: 2
simple_rnn_7/while/Identity_1�
simple_rnn_7/while/Identity_2Identitysimple_rnn_7/while/add:z:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/Identity_2�
simple_rnn_7/while/Identity_3IdentityGsimple_rnn_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/Identity_3�
simple_rnn_7/while/Identity_4Identity-simple_rnn_7/while/simple_rnn_cell_8/Tanh:y:0*
T0*
_output_shapes

:_2
simple_rnn_7/while/Identity_4"C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0"G
simple_rnn_7_while_identity_1&simple_rnn_7/while/Identity_1:output:0"G
simple_rnn_7_while_identity_2&simple_rnn_7/while/Identity_2:output:0"G
simple_rnn_7_while_identity_3&simple_rnn_7/while/Identity_3:output:0"G
simple_rnn_7_while_identity_4&simple_rnn_7/while/Identity_4:output:0"`
-simple_rnn_7_while_simple_rnn_7_strided_slice/simple_rnn_7_while_simple_rnn_7_strided_slice_0"�
Dsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resourceFsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"�
Esimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resourceGsimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"�
Csimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resourceEsimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resource_0"�
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensormsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0*5
_input_shapes$
": : : : :_: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_sequential_11_layer_call_fn_60761

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:d_*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_11_layer_call_and_return_conditional_losses_602102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������d:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�3
�
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_59869

inputs
simple_rnn_cell_8_59791
simple_rnn_cell_8_59793
simple_rnn_cell_8_59795
simple_rnn_cell_8_59797
identity��AssignVariableOp�)simple_rnn_cell_8/StatefulPartitionedCall�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
*
shrink_axis_mask2
strided_slice_1�
)simple_rnn_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0simple_rnn_cell_8_59791simple_rnn_cell_8_59793simple_rnn_cell_8_59795simple_rnn_cell_8_59797*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:_:_*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_593932+
)simple_rnn_cell_8/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
timex
ReadVariableOpReadVariableOpsimple_rnn_cell_8_59791*
_output_shapes

:_*
dtype02
ReadVariableOp
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_8_59793simple_rnn_cell_8_59795simple_rnn_cell_8_59797*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :_: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_59806*
condR
while_cond_59805*/
output_shapes
: : : : :_: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������_*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:_*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������_2
transpose_1�
AssignVariableOpAssignVariableOpsimple_rnn_cell_8_59791while:output:4^ReadVariableOp*^simple_rnn_cell_8/StatefulPartitionedCall*
_output_shapes
 *
dtype02
AssignVariableOp�
IdentityIdentitytranspose_1:y:0^AssignVariableOp*^simple_rnn_cell_8/StatefulPartitionedCall^while*
T0*+
_output_shapes
:���������_2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������
::::2$
AssignVariableOpAssignVariableOp2V
)simple_rnn_cell_8/StatefulPartitionedCall)simple_rnn_cell_8/StatefulPartitionedCall2
whilewhile:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
while_cond_60852
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_60852___redundant_placeholder03
/while_while_cond_60852___redundant_placeholder13
/while_while_cond_60852___redundant_placeholder23
/while_while_cond_60852___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*7
_input_shapes&
$: : : : :_: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
:
�
�
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_61398

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
_*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:_*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2	
BiasAdd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:__*
dtype02
MatMul_1/ReadVariableOpr
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:_2

MatMul_1b
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*
_output_shapes

:_2
addF
TanhTanhadd:z:0*
T0*
_output_shapes

:_2
TanhS
IdentityIdentityTanh:y:0*
T0*
_output_shapes

:_2

IdentityW

Identity_1IdentityTanh:y:0*
T0*
_output_shapes

:_2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*3
_input_shapes"
 :
:_::::F B

_output_shapes

:

 
_user_specified_nameinputs:HD

_output_shapes

:_
"
_user_specified_name
states/0
�
�
H__inference_sequential_11_layer_call_and_return_conditional_losses_60210

inputs
embedding_11_60197
simple_rnn_7_60200
simple_rnn_7_60202
simple_rnn_7_60204
simple_rnn_7_60206
identity��$embedding_11/StatefulPartitionedCall�$simple_rnn_7/StatefulPartitionedCall�
lambda_11/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_lambda_11_layer_call_and_return_conditional_losses_598912
lambda_11/PartitionedCall�
$embedding_11/StatefulPartitionedCallStatefulPartitionedCall"lambda_11/PartitionedCall:output:0embedding_11_60197*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:d
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_11_layer_call_and_return_conditional_losses_599212&
$embedding_11/StatefulPartitionedCall�
$simple_rnn_7/StatefulPartitionedCallStatefulPartitionedCall-embedding_11/StatefulPartitionedCall:output:0simple_rnn_7_60200simple_rnn_7_60202simple_rnn_7_60204simple_rnn_7_60206*
Tin	
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:d_*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_600342&
$simple_rnn_7/StatefulPartitionedCall�
IdentityIdentity-simple_rnn_7/StatefulPartitionedCall:output:0%^embedding_11/StatefulPartitionedCall%^simple_rnn_7/StatefulPartitionedCall*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*5
_input_shapes$
":d:::::2L
$embedding_11/StatefulPartitionedCall$embedding_11/StatefulPartitionedCall2L
$simple_rnn_7/StatefulPartitionedCall$simple_rnn_7/StatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�<
�
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_60136

inputs4
0simple_rnn_cell_8_matmul_readvariableop_resource5
1simple_rnn_cell_8_biasadd_readvariableop_resource6
2simple_rnn_cell_8_matmul_1_readvariableop_resource8
4simple_rnn_cell_8_matmul_1_readvariableop_1_resource
identity��AssignVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permq
	transpose	Transposeinputstranspose/perm:output:0*
T0*"
_output_shapes
:d
2
	transposec
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d      
   2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
*
shrink_axis_mask2
strided_slice_1�
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

:
_*
dtype02)
'simple_rnn_cell_8/MatMul/ReadVariableOp�
simple_rnn_cell_8/MatMulMatMulstrided_slice_1:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/MatMul�
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype02*
(simple_rnn_cell_8/BiasAdd/ReadVariableOp�
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/BiasAdd�
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02+
)simple_rnn_cell_8/MatMul_1/ReadVariableOp�
+simple_rnn_cell_8/MatMul_1/ReadVariableOp_1ReadVariableOp4simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02-
+simple_rnn_cell_8/MatMul_1/ReadVariableOp_1�
simple_rnn_cell_8/MatMul_1MatMul1simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:03simple_rnn_cell_8/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/MatMul_1�
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/add|
simple_rnn_cell_8/TanhTanhsimple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time�
ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02
ReadVariableOp
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource4simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :_: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_60070*
condR
while_cond_60069*/
output_shapes
: : : : :_: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:d_*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:_*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*"
_output_shapes
:d_2
transpose_1�
AssignVariableOpAssignVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resourcewhile:output:4^ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpy
IdentityIdentitytranspose_1:y:0^AssignVariableOp^while*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*1
_input_shapes 
:d
::::2$
AssignVariableOpAssignVariableOp2
whilewhile:J F
"
_output_shapes
:d

 
_user_specified_nameinputs
�
�
while_cond_59805
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_59805___redundant_placeholder03
/while_while_cond_59805___redundant_placeholder13
/while_while_cond_59805___redundant_placeholder23
/while_while_cond_59805___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*7
_input_shapes&
$: : : : :_: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
:
�
`
D__inference_lambda_11_layer_call_and_return_conditional_losses_59898

inputs
identityS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
add/yX
addAddV2inputsadd/y:output:0*
T0*"
_output_shapes
:d2
addo
SqueezeSqueezeadd:z:0*
T0*
_output_shapes

:d*
squeeze_dims

���������2	
Squeeze[
IdentityIdentitySqueeze:output:0*
T0*
_output_shapes

:d2

Identity"
identityIdentity:output:0*!
_input_shapes
:d:J F
"
_output_shapes
:d
 
_user_specified_nameinputs
�)
�
while_body_61083
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_8_matmul_readvariableop_resource;
7while_simple_rnn_cell_8_biasadd_readvariableop_resource<
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource��
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:
_*
dtype02/
-while/simple_rnn_cell_8/MatMul/ReadVariableOp�
while/simple_rnn_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2 
while/simple_rnn_cell_8/MatMul�
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:_*
dtype020
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2!
while/simple_rnn_cell_8/BiasAdd�
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:__*
dtype021
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_8/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:_2"
 while/simple_rnn_cell_8/MatMul_1�
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2
while/simple_rnn_cell_8/add�
while/simple_rnn_cell_8/TanhTanhwhile/simple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2
while/simple_rnn_cell_8/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_8/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3{
while/Identity_4Identity while/simple_rnn_cell_8/Tanh:y:0*
T0*
_output_shapes

:_2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*5
_input_shapes$
": : : : :_: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
: 
�
�
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_61296

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource&
"matmul_1_readvariableop_1_resource
identity

identity_1��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
_*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:_*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2	
BiasAddu
MatMul_1/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1/ReadVariableOp_1ReadVariableOp"matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02
MatMul_1/ReadVariableOp_1�
MatMul_1BatchMatMulV2MatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:2

MatMul_1[
addAddV2BiasAdd:output:0MatMul_1:output:0*
T0*
_output_shapes
:2
add@
TanhTanhadd:z:0*
T0*
_output_shapes
:2
TanhM
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:2

IdentityQ

Identity_1IdentityTanh:y:0*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*-
_input_shapes
:
:::::F B

_output_shapes

:

 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
�
�
-__inference_sequential_11_layer_call_fn_60776

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:d_*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_11_layer_call_and_return_conditional_losses_602422
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������d:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
,__inference_simple_rnn_7_layer_call_fn_61047
inputs_0
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������_*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_598692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������_2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������
::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������

"
_user_specified_name
inputs/0
�
E
)__inference_lambda_11_layer_call_fn_60800

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_lambda_11_layer_call_and_return_conditional_losses_598982
PartitionedCallc
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:d2

Identity"
identityIdentity:output:0*!
_input_shapes
:d:J F
"
_output_shapes
:d
 
_user_specified_nameinputs
�
�
,__inference_simple_rnn_7_layer_call_fn_61277

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:d_*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_601362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*1
_input_shapes 
:d
::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:d

 
_user_specified_nameinputs
�)
�
while_body_60955
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_8_matmul_readvariableop_resource;
7while_simple_rnn_cell_8_biasadd_readvariableop_resource<
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource��
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:
_*
dtype02/
-while/simple_rnn_cell_8/MatMul/ReadVariableOp�
while/simple_rnn_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2 
while/simple_rnn_cell_8/MatMul�
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:_*
dtype020
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2!
while/simple_rnn_cell_8/BiasAdd�
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:__*
dtype021
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_8/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:_2"
 while/simple_rnn_cell_8/MatMul_1�
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2
while/simple_rnn_cell_8/add�
while/simple_rnn_cell_8/TanhTanhwhile/simple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2
while/simple_rnn_cell_8/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_8/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3{
while/Identity_4Identity while/simple_rnn_cell_8/Tanh:y:0*
T0*
_output_shapes

:_2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*5
_input_shapes$
": : : : :_: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
: 
�<
�
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_61021
inputs_04
0simple_rnn_cell_8_matmul_readvariableop_resource5
1simple_rnn_cell_8_biasadd_readvariableop_resource6
2simple_rnn_cell_8_matmul_1_readvariableop_resource8
4simple_rnn_cell_8_matmul_1_readvariableop_1_resource
identity��AssignVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*+
_output_shapes
:���������
2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
*
shrink_axis_mask2
strided_slice_1�
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

:
_*
dtype02)
'simple_rnn_cell_8/MatMul/ReadVariableOp�
simple_rnn_cell_8/MatMulMatMulstrided_slice_1:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/MatMul�
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype02*
(simple_rnn_cell_8/BiasAdd/ReadVariableOp�
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/BiasAdd�
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02+
)simple_rnn_cell_8/MatMul_1/ReadVariableOp�
+simple_rnn_cell_8/MatMul_1/ReadVariableOp_1ReadVariableOp4simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02-
+simple_rnn_cell_8/MatMul_1/ReadVariableOp_1�
simple_rnn_cell_8/MatMul_1MatMul1simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:03simple_rnn_cell_8/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/MatMul_1�
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/add|
simple_rnn_cell_8/TanhTanhsimple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time�
ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02
ReadVariableOp
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource4simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :_: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_60955*
condR
while_cond_60954*/
output_shapes
: : : : :_: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������_*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:_*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������_2
transpose_1�
AssignVariableOpAssignVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resourcewhile:output:4^ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp�
IdentityIdentitytranspose_1:y:0^AssignVariableOp^while*
T0*+
_output_shapes
:���������_2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������
::::2$
AssignVariableOpAssignVariableOp2
whilewhile:U Q
+
_output_shapes
:���������

"
_user_specified_name
inputs/0
�)
�
while_body_59968
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_8_matmul_readvariableop_resource;
7while_simple_rnn_cell_8_biasadd_readvariableop_resource<
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource��
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:
_*
dtype02/
-while/simple_rnn_cell_8/MatMul/ReadVariableOp�
while/simple_rnn_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2 
while/simple_rnn_cell_8/MatMul�
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:_*
dtype020
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2!
while/simple_rnn_cell_8/BiasAdd�
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:__*
dtype021
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_8/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:_2"
 while/simple_rnn_cell_8/MatMul_1�
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2
while/simple_rnn_cell_8/add�
while/simple_rnn_cell_8/TanhTanhwhile/simple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2
while/simple_rnn_cell_8/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_8/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3{
while/Identity_4Identity while/simple_rnn_cell_8/Tanh:y:0*
T0*
_output_shapes

:_2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*5
_input_shapes$
": : : : :_: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
: 
�
E
)__inference_lambda_11_layer_call_fn_60795

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_lambda_11_layer_call_and_return_conditional_losses_598912
PartitionedCallc
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:d2

Identity"
identityIdentity:output:0*!
_input_shapes
:d:J F
"
_output_shapes
:d
 
_user_specified_nameinputs
�
�
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_61415

inputs
states_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
_*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:_*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2	
BiasAdd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:__*
dtype02
MatMul_1/ReadVariableOpr
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:_2

MatMul_1b
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*
_output_shapes

:_2
addF
TanhTanhadd:z:0*
T0*
_output_shapes

:_2
TanhS
IdentityIdentityTanh:y:0*
T0*
_output_shapes

:_2

IdentityW

Identity_1IdentityTanh:y:0*
T0*
_output_shapes

:_2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*3
_input_shapes"
 :
:_::::F B

_output_shapes

:

 
_user_specified_nameinputs:HD

_output_shapes

:_
"
_user_specified_name
states/0
�h
�
 __inference__wrapped_model_59338
input_125
1sequential_11_embedding_11_embedding_lookup_59234O
Ksequential_11_simple_rnn_7_simple_rnn_cell_8_matmul_readvariableop_resourceP
Lsequential_11_simple_rnn_7_simple_rnn_cell_8_biasadd_readvariableop_resourceQ
Msequential_11_simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resourceS
Osequential_11_simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_1_resource
identity��+sequential_11/simple_rnn_7/AssignVariableOp� sequential_11/simple_rnn_7/while�
sequential_11/lambda_11/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sequential_11/lambda_11/add/y�
sequential_11/lambda_11/addAddV2input_12&sequential_11/lambda_11/add/y:output:0*
T0*"
_output_shapes
:d2
sequential_11/lambda_11/add�
sequential_11/lambda_11/SqueezeSqueezesequential_11/lambda_11/add:z:0*
T0*
_output_shapes

:d*
squeeze_dims

���������2!
sequential_11/lambda_11/Squeeze�
sequential_11/embedding_11/CastCast(sequential_11/lambda_11/Squeeze:output:0*

DstT0*

SrcT0*
_output_shapes

:d2!
sequential_11/embedding_11/Cast�
+sequential_11/embedding_11/embedding_lookupResourceGather1sequential_11_embedding_11_embedding_lookup_59234#sequential_11/embedding_11/Cast:y:0*
Tindices0*D
_class:
86loc:@sequential_11/embedding_11/embedding_lookup/59234*"
_output_shapes
:d
*
dtype02-
+sequential_11/embedding_11/embedding_lookup�
4sequential_11/embedding_11/embedding_lookup/IdentityIdentity4sequential_11/embedding_11/embedding_lookup:output:0*
T0*D
_class:
86loc:@sequential_11/embedding_11/embedding_lookup/59234*"
_output_shapes
:d
26
4sequential_11/embedding_11/embedding_lookup/Identity�
6sequential_11/embedding_11/embedding_lookup/Identity_1Identity=sequential_11/embedding_11/embedding_lookup/Identity:output:0*
T0*"
_output_shapes
:d
28
6sequential_11/embedding_11/embedding_lookup/Identity_1�
)sequential_11/simple_rnn_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)sequential_11/simple_rnn_7/transpose/perm�
$sequential_11/simple_rnn_7/transpose	Transpose?sequential_11/embedding_11/embedding_lookup/Identity_1:output:02sequential_11/simple_rnn_7/transpose/perm:output:0*
T0*"
_output_shapes
:d
2&
$sequential_11/simple_rnn_7/transpose�
 sequential_11/simple_rnn_7/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d      
   2"
 sequential_11/simple_rnn_7/Shape�
.sequential_11/simple_rnn_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_11/simple_rnn_7/strided_slice/stack�
0sequential_11/simple_rnn_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_11/simple_rnn_7/strided_slice/stack_1�
0sequential_11/simple_rnn_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential_11/simple_rnn_7/strided_slice/stack_2�
(sequential_11/simple_rnn_7/strided_sliceStridedSlice)sequential_11/simple_rnn_7/Shape:output:07sequential_11/simple_rnn_7/strided_slice/stack:output:09sequential_11/simple_rnn_7/strided_slice/stack_1:output:09sequential_11/simple_rnn_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(sequential_11/simple_rnn_7/strided_slice�
6sequential_11/simple_rnn_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������28
6sequential_11/simple_rnn_7/TensorArrayV2/element_shape�
(sequential_11/simple_rnn_7/TensorArrayV2TensorListReserve?sequential_11/simple_rnn_7/TensorArrayV2/element_shape:output:01sequential_11/simple_rnn_7/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02*
(sequential_11/simple_rnn_7/TensorArrayV2�
Psequential_11/simple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   2R
Psequential_11/simple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape�
Bsequential_11/simple_rnn_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_11/simple_rnn_7/transpose:y:0Ysequential_11/simple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02D
Bsequential_11/simple_rnn_7/TensorArrayUnstack/TensorListFromTensor�
0sequential_11/simple_rnn_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_11/simple_rnn_7/strided_slice_1/stack�
2sequential_11/simple_rnn_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_11/simple_rnn_7/strided_slice_1/stack_1�
2sequential_11/simple_rnn_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_11/simple_rnn_7/strided_slice_1/stack_2�
*sequential_11/simple_rnn_7/strided_slice_1StridedSlice(sequential_11/simple_rnn_7/transpose:y:09sequential_11/simple_rnn_7/strided_slice_1/stack:output:0;sequential_11/simple_rnn_7/strided_slice_1/stack_1:output:0;sequential_11/simple_rnn_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
*
shrink_axis_mask2,
*sequential_11/simple_rnn_7/strided_slice_1�
Bsequential_11/simple_rnn_7/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOpKsequential_11_simple_rnn_7_simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

:
_*
dtype02D
Bsequential_11/simple_rnn_7/simple_rnn_cell_8/MatMul/ReadVariableOp�
3sequential_11/simple_rnn_7/simple_rnn_cell_8/MatMulMatMul3sequential_11/simple_rnn_7/strided_slice_1:output:0Jsequential_11/simple_rnn_7/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_25
3sequential_11/simple_rnn_7/simple_rnn_cell_8/MatMul�
Csequential_11/simple_rnn_7/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOpLsequential_11_simple_rnn_7_simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype02E
Csequential_11/simple_rnn_7/simple_rnn_cell_8/BiasAdd/ReadVariableOp�
4sequential_11/simple_rnn_7/simple_rnn_cell_8/BiasAddBiasAdd=sequential_11/simple_rnn_7/simple_rnn_cell_8/MatMul:product:0Ksequential_11/simple_rnn_7/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_26
4sequential_11/simple_rnn_7/simple_rnn_cell_8/BiasAdd�
Dsequential_11/simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOpMsequential_11_simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02F
Dsequential_11/simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp�
Fsequential_11/simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp_1ReadVariableOpOsequential_11_simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02H
Fsequential_11/simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp_1�
5sequential_11/simple_rnn_7/simple_rnn_cell_8/MatMul_1MatMulLsequential_11/simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0Nsequential_11/simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:_27
5sequential_11/simple_rnn_7/simple_rnn_cell_8/MatMul_1�
0sequential_11/simple_rnn_7/simple_rnn_cell_8/addAddV2=sequential_11/simple_rnn_7/simple_rnn_cell_8/BiasAdd:output:0?sequential_11/simple_rnn_7/simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_22
0sequential_11/simple_rnn_7/simple_rnn_cell_8/add�
1sequential_11/simple_rnn_7/simple_rnn_cell_8/TanhTanh4sequential_11/simple_rnn_7/simple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_23
1sequential_11/simple_rnn_7/simple_rnn_cell_8/Tanh�
8sequential_11/simple_rnn_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2:
8sequential_11/simple_rnn_7/TensorArrayV2_1/element_shape�
*sequential_11/simple_rnn_7/TensorArrayV2_1TensorListReserveAsequential_11/simple_rnn_7/TensorArrayV2_1/element_shape:output:01sequential_11/simple_rnn_7/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02,
*sequential_11/simple_rnn_7/TensorArrayV2_1�
sequential_11/simple_rnn_7/timeConst*
_output_shapes
: *
dtype0*
value	B : 2!
sequential_11/simple_rnn_7/time�
)sequential_11/simple_rnn_7/ReadVariableOpReadVariableOpMsequential_11_simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02+
)sequential_11/simple_rnn_7/ReadVariableOp�
3sequential_11/simple_rnn_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������25
3sequential_11/simple_rnn_7/while/maximum_iterations�
-sequential_11/simple_rnn_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_11/simple_rnn_7/while/loop_counter�
 sequential_11/simple_rnn_7/whileWhile6sequential_11/simple_rnn_7/while/loop_counter:output:0<sequential_11/simple_rnn_7/while/maximum_iterations:output:0(sequential_11/simple_rnn_7/time:output:03sequential_11/simple_rnn_7/TensorArrayV2_1:handle:01sequential_11/simple_rnn_7/ReadVariableOp:value:01sequential_11/simple_rnn_7/strided_slice:output:0Rsequential_11/simple_rnn_7/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ksequential_11_simple_rnn_7_simple_rnn_cell_8_matmul_readvariableop_resourceLsequential_11_simple_rnn_7_simple_rnn_cell_8_biasadd_readvariableop_resourceOsequential_11_simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :_: : : : : *%
_read_only_resource_inputs
	*7
body/R-
+sequential_11_simple_rnn_7_while_body_59272*7
cond/R-
+sequential_11_simple_rnn_7_while_cond_59271*/
output_shapes
: : : : :_: : : : : *
parallel_iterations 2"
 sequential_11/simple_rnn_7/while�
Ksequential_11/simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2M
Ksequential_11/simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape�
=sequential_11/simple_rnn_7/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_11/simple_rnn_7/while:output:3Tsequential_11/simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:d_*
element_dtype02?
=sequential_11/simple_rnn_7/TensorArrayV2Stack/TensorListStack�
0sequential_11/simple_rnn_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������22
0sequential_11/simple_rnn_7/strided_slice_2/stack�
2sequential_11/simple_rnn_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_11/simple_rnn_7/strided_slice_2/stack_1�
2sequential_11/simple_rnn_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_11/simple_rnn_7/strided_slice_2/stack_2�
*sequential_11/simple_rnn_7/strided_slice_2StridedSliceFsequential_11/simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:09sequential_11/simple_rnn_7/strided_slice_2/stack:output:0;sequential_11/simple_rnn_7/strided_slice_2/stack_1:output:0;sequential_11/simple_rnn_7/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:_*
shrink_axis_mask2,
*sequential_11/simple_rnn_7/strided_slice_2�
+sequential_11/simple_rnn_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2-
+sequential_11/simple_rnn_7/transpose_1/perm�
&sequential_11/simple_rnn_7/transpose_1	TransposeFsequential_11/simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:04sequential_11/simple_rnn_7/transpose_1/perm:output:0*
T0*"
_output_shapes
:d_2(
&sequential_11/simple_rnn_7/transpose_1�
+sequential_11/simple_rnn_7/AssignVariableOpAssignVariableOpMsequential_11_simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resource)sequential_11/simple_rnn_7/while:output:4*^sequential_11/simple_rnn_7/ReadVariableOpE^sequential_11/simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
_output_shapes
 *
dtype02-
+sequential_11/simple_rnn_7/AssignVariableOp�
IdentityIdentity*sequential_11/simple_rnn_7/transpose_1:y:0,^sequential_11/simple_rnn_7/AssignVariableOp!^sequential_11/simple_rnn_7/while*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*5
_input_shapes$
":d:::::2Z
+sequential_11/simple_rnn_7/AssignVariableOp+sequential_11/simple_rnn_7/AssignVariableOp2D
 sequential_11/simple_rnn_7/while sequential_11/simple_rnn_7/while:U Q
+
_output_shapes
:���������d
"
_user_specified_name
input_12
�
�
while_cond_61082
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_61082___redundant_placeholder03
/while_while_cond_61082___redundant_placeholder13
/while_while_cond_61082___redundant_placeholder23
/while_while_cond_61082___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*7
_input_shapes&
$: : : : :_: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
:
�
�
G__inference_embedding_11_layer_call_and_return_conditional_losses_60810

inputs
embedding_lookup_60804
identity�T
CastCastinputs*

DstT0*

SrcT0*
_output_shapes

:d2
Cast�
embedding_lookupResourceGatherembedding_lookup_60804Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/60804*"
_output_shapes
:d
*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/60804*"
_output_shapes
:d
2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*"
_output_shapes
:d
2
embedding_lookup/Identity_1s
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*"
_output_shapes
:d
2

Identity"
identityIdentity:output:0*!
_input_shapes
:d::F B

_output_shapes

:d
 
_user_specified_nameinputs
�
�
!__inference__traced_restore_61506
file_prefix,
(assignvariableop_embedding_11_embeddings<
8assignvariableop_1_simple_rnn_7_simple_rnn_cell_8_kernelF
Bassignvariableop_2_simple_rnn_7_simple_rnn_cell_8_recurrent_kernel:
6assignvariableop_3_simple_rnn_7_simple_rnn_cell_8_bias,
(assignvariableop_4_simple_rnn_7_variable

identity_6��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-1/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp(assignvariableop_embedding_11_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp8assignvariableop_1_simple_rnn_7_simple_rnn_cell_8_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpBassignvariableop_2_simple_rnn_7_simple_rnn_cell_8_recurrent_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp6assignvariableop_3_simple_rnn_7_simple_rnn_cell_8_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp(assignvariableop_4_simple_rnn_7_variableIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5�

Identity_6IdentityIdentity_5:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4*
T0*
_output_shapes
: 2

Identity_6"!

identity_6Identity_6:output:0*)
_input_shapes
: :::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_4:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�)
�
while_body_60070
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_8_matmul_readvariableop_resource;
7while_simple_rnn_cell_8_biasadd_readvariableop_resource<
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource��
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:
_*
dtype02/
-while/simple_rnn_cell_8/MatMul/ReadVariableOp�
while/simple_rnn_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2 
while/simple_rnn_cell_8/MatMul�
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:_*
dtype020
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2!
while/simple_rnn_cell_8/BiasAdd�
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:__*
dtype021
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_8/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:_2"
 while/simple_rnn_cell_8/MatMul_1�
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2
while/simple_rnn_cell_8/add�
while/simple_rnn_cell_8/TanhTanhwhile/simple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2
while/simple_rnn_cell_8/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_8/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3{
while/Identity_4Identity while/simple_rnn_cell_8/Tanh:y:0*
T0*
_output_shapes

:_2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*5
_input_shapes$
": : : : :_: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
: 
�"
�
while_body_59806
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_simple_rnn_cell_8_59828_0#
while_simple_rnn_cell_8_59830_0#
while_simple_rnn_cell_8_59832_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_simple_rnn_cell_8_59828!
while_simple_rnn_cell_8_59830!
while_simple_rnn_cell_8_59832��/while/simple_rnn_cell_8/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
/while/simple_rnn_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_8_59828_0while_simple_rnn_cell_8_59830_0while_simple_rnn_cell_8_59832_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:_:_*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_5949521
/while/simple_rnn_cell_8/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_8/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:00^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations0^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:00^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity8while/simple_rnn_cell_8/StatefulPartitionedCall:output:10^while/simple_rnn_cell_8/StatefulPartitionedCall*
T0*
_output_shapes

:_2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_8_59828while_simple_rnn_cell_8_59828_0"@
while_simple_rnn_cell_8_59830while_simple_rnn_cell_8_59830_0"@
while_simple_rnn_cell_8_59832while_simple_rnn_cell_8_59832_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*5
_input_shapes$
": : : : :_: : :::2b
/while/simple_rnn_cell_8/StatefulPartitionedCall/while/simple_rnn_cell_8/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_sequential_11_layer_call_and_return_conditional_losses_60242

inputs
embedding_11_60229
simple_rnn_7_60232
simple_rnn_7_60234
simple_rnn_7_60236
simple_rnn_7_60238
identity��$embedding_11/StatefulPartitionedCall�$simple_rnn_7/StatefulPartitionedCall�
lambda_11/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_lambda_11_layer_call_and_return_conditional_losses_598982
lambda_11/PartitionedCall�
$embedding_11/StatefulPartitionedCallStatefulPartitionedCall"lambda_11/PartitionedCall:output:0embedding_11_60229*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:d
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_11_layer_call_and_return_conditional_losses_599212&
$embedding_11/StatefulPartitionedCall�
$simple_rnn_7/StatefulPartitionedCallStatefulPartitionedCall-embedding_11/StatefulPartitionedCall:output:0simple_rnn_7_60232simple_rnn_7_60234simple_rnn_7_60236simple_rnn_7_60238*
Tin	
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:d_*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_601362&
$simple_rnn_7/StatefulPartitionedCall�
IdentityIdentity-simple_rnn_7/StatefulPartitionedCall:output:0%^embedding_11/StatefulPartitionedCall%^simple_rnn_7/StatefulPartitionedCall*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*5
_input_shapes$
":d:::::2L
$embedding_11/StatefulPartitionedCall$embedding_11/StatefulPartitionedCall2L
$simple_rnn_7/StatefulPartitionedCall$simple_rnn_7/StatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_61370

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource&
"matmul_1_readvariableop_1_resource
identity

identity_1��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
_*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:_*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2	
BiasAdds
MatMul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1/ReadVariableOp_1ReadVariableOp"matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02
MatMul_1/ReadVariableOp_1�
MatMul_1BatchMatMulV2MatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:2

MatMul_1[
addAddV2BiasAdd:output:0MatMul_1:output:0*
T0*
_output_shapes
:2
add@
TanhTanhadd:z:0*
T0*
_output_shapes
:2
TanhM
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:2

IdentityQ

Identity_1IdentityTanh:y:0*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*-
_input_shapes
:
:::::F B

_output_shapes

:

 
_user_specified_nameinputs:&"
 
_user_specified_namestates
�6
�	
simple_rnn_7_while_body_606806
2simple_rnn_7_while_simple_rnn_7_while_loop_counter<
8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations"
simple_rnn_7_while_placeholder$
 simple_rnn_7_while_placeholder_1$
 simple_rnn_7_while_placeholder_23
/simple_rnn_7_while_simple_rnn_7_strided_slice_0q
msimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0I
Esimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resource_0J
Fsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0K
Gsimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
simple_rnn_7_while_identity!
simple_rnn_7_while_identity_1!
simple_rnn_7_while_identity_2!
simple_rnn_7_while_identity_3!
simple_rnn_7_while_identity_41
-simple_rnn_7_while_simple_rnn_7_strided_sliceo
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensorG
Csimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resourceH
Dsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resourceI
Esimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource��
Dsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   2F
Dsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape�
6simple_rnn_7/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_7_while_placeholderMsimple_rnn_7/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
*
element_dtype028
6simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem�
:simple_rnn_7/while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:
_*
dtype02<
:simple_rnn_7/while/simple_rnn_cell_8/MatMul/ReadVariableOp�
+simple_rnn_7/while/simple_rnn_cell_8/MatMulMatMul=simple_rnn_7/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_7/while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2-
+simple_rnn_7/while/simple_rnn_cell_8/MatMul�
;simple_rnn_7/while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:_*
dtype02=
;simple_rnn_7/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp�
,simple_rnn_7/while/simple_rnn_cell_8/BiasAddBiasAdd5simple_rnn_7/while/simple_rnn_cell_8/MatMul:product:0Csimple_rnn_7/while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2.
,simple_rnn_7/while/simple_rnn_cell_8/BiasAdd�
<simple_rnn_7/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:__*
dtype02>
<simple_rnn_7/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp�
-simple_rnn_7/while/simple_rnn_cell_8/MatMul_1MatMul simple_rnn_7_while_placeholder_2Dsimple_rnn_7/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:_2/
-simple_rnn_7/while/simple_rnn_cell_8/MatMul_1�
(simple_rnn_7/while/simple_rnn_cell_8/addAddV25simple_rnn_7/while/simple_rnn_cell_8/BiasAdd:output:07simple_rnn_7/while/simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2*
(simple_rnn_7/while/simple_rnn_cell_8/add�
)simple_rnn_7/while/simple_rnn_cell_8/TanhTanh,simple_rnn_7/while/simple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2+
)simple_rnn_7/while/simple_rnn_cell_8/Tanh�
7simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_7_while_placeholder_1simple_rnn_7_while_placeholder-simple_rnn_7/while/simple_rnn_cell_8/Tanh:y:0*
_output_shapes
: *
element_dtype029
7simple_rnn_7/while/TensorArrayV2Write/TensorListSetItemv
simple_rnn_7/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_7/while/add/y�
simple_rnn_7/while/addAddV2simple_rnn_7_while_placeholder!simple_rnn_7/while/add/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/addz
simple_rnn_7/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
simple_rnn_7/while/add_1/y�
simple_rnn_7/while/add_1AddV22simple_rnn_7_while_simple_rnn_7_while_loop_counter#simple_rnn_7/while/add_1/y:output:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/add_1�
simple_rnn_7/while/IdentityIdentitysimple_rnn_7/while/add_1:z:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/Identity�
simple_rnn_7/while/Identity_1Identity8simple_rnn_7_while_simple_rnn_7_while_maximum_iterations*
T0*
_output_shapes
: 2
simple_rnn_7/while/Identity_1�
simple_rnn_7/while/Identity_2Identitysimple_rnn_7/while/add:z:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/Identity_2�
simple_rnn_7/while/Identity_3IdentityGsimple_rnn_7/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
simple_rnn_7/while/Identity_3�
simple_rnn_7/while/Identity_4Identity-simple_rnn_7/while/simple_rnn_cell_8/Tanh:y:0*
T0*
_output_shapes

:_2
simple_rnn_7/while/Identity_4"C
simple_rnn_7_while_identity$simple_rnn_7/while/Identity:output:0"G
simple_rnn_7_while_identity_1&simple_rnn_7/while/Identity_1:output:0"G
simple_rnn_7_while_identity_2&simple_rnn_7/while/Identity_2:output:0"G
simple_rnn_7_while_identity_3&simple_rnn_7/while/Identity_3:output:0"G
simple_rnn_7_while_identity_4&simple_rnn_7/while/Identity_4:output:0"`
-simple_rnn_7_while_simple_rnn_7_strided_slice/simple_rnn_7_while_simple_rnn_7_strided_slice_0"�
Dsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resourceFsimple_rnn_7_while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"�
Esimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resourceGsimple_rnn_7_while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"�
Csimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resourceEsimple_rnn_7_while_simple_rnn_cell_8_matmul_readvariableop_resource_0"�
ksimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensormsimple_rnn_7_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_7_tensorarrayunstack_tensorlistfromtensor_0*5
_input_shapes$
": : : : :_: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
: 
�<
�
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_61251

inputs4
0simple_rnn_cell_8_matmul_readvariableop_resource5
1simple_rnn_cell_8_biasadd_readvariableop_resource6
2simple_rnn_cell_8_matmul_1_readvariableop_resource8
4simple_rnn_cell_8_matmul_1_readvariableop_1_resource
identity��AssignVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permq
	transpose	Transposeinputstranspose/perm:output:0*
T0*"
_output_shapes
:d
2
	transposec
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d      
   2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
*
shrink_axis_mask2
strided_slice_1�
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

:
_*
dtype02)
'simple_rnn_cell_8/MatMul/ReadVariableOp�
simple_rnn_cell_8/MatMulMatMulstrided_slice_1:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/MatMul�
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype02*
(simple_rnn_cell_8/BiasAdd/ReadVariableOp�
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/BiasAdd�
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02+
)simple_rnn_cell_8/MatMul_1/ReadVariableOp�
+simple_rnn_cell_8/MatMul_1/ReadVariableOp_1ReadVariableOp4simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02-
+simple_rnn_cell_8/MatMul_1/ReadVariableOp_1�
simple_rnn_cell_8/MatMul_1MatMul1simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:03simple_rnn_cell_8/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/MatMul_1�
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/add|
simple_rnn_cell_8/TanhTanhsimple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time�
ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02
ReadVariableOp
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource4simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :_: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_61185*
condR
while_cond_61184*/
output_shapes
: : : : :_: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:d_*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:_*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*"
_output_shapes
:d_2
transpose_1�
AssignVariableOpAssignVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resourcewhile:output:4^ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpy
IdentityIdentitytranspose_1:y:0^AssignVariableOp^while*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*1
_input_shapes 
:d
::::2$
AssignVariableOpAssignVariableOp2
whilewhile:J F
"
_output_shapes
:d

 
_user_specified_nameinputs
�
�
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_59375

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource&
"matmul_1_readvariableop_1_resource
identity

identity_1��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
_*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:_*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2	
BiasAddy
MatMul_1/ReadVariableOpReadVariableOpstates*
_output_shapes

:_*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1/ReadVariableOp_1ReadVariableOp"matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02
MatMul_1/ReadVariableOp_1�
MatMul_1MatMulMatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:_2

MatMul_1b
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*
_output_shapes

:_2
addF
TanhTanhadd:z:0*
T0*
_output_shapes

:_2
TanhS
IdentityIdentityTanh:y:0*
T0*
_output_shapes

:_2

IdentityW

Identity_1IdentityTanh:y:0*
T0*
_output_shapes

:_2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*-
_input_shapes
:
:::::F B

_output_shapes

:

 
_user_specified_nameinputs:&"
 
_user_specified_namestates
�U
�
H__inference_sequential_11_layer_call_and_return_conditional_losses_60383
input_12'
#embedding_11_embedding_lookup_60279A
=simple_rnn_7_simple_rnn_cell_8_matmul_readvariableop_resourceB
>simple_rnn_7_simple_rnn_cell_8_biasadd_readvariableop_resourceC
?simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resourceE
Asimple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_1_resource
identity��simple_rnn_7/AssignVariableOp�simple_rnn_7/whileg
lambda_11/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
lambda_11/add/yx
lambda_11/addAddV2input_12lambda_11/add/y:output:0*
T0*"
_output_shapes
:d2
lambda_11/add�
lambda_11/SqueezeSqueezelambda_11/add:z:0*
T0*
_output_shapes

:d*
squeeze_dims

���������2
lambda_11/Squeeze�
embedding_11/CastCastlambda_11/Squeeze:output:0*

DstT0*

SrcT0*
_output_shapes

:d2
embedding_11/Cast�
embedding_11/embedding_lookupResourceGather#embedding_11_embedding_lookup_60279embedding_11/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_11/embedding_lookup/60279*"
_output_shapes
:d
*
dtype02
embedding_11/embedding_lookup�
&embedding_11/embedding_lookup/IdentityIdentity&embedding_11/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_11/embedding_lookup/60279*"
_output_shapes
:d
2(
&embedding_11/embedding_lookup/Identity�
(embedding_11/embedding_lookup/Identity_1Identity/embedding_11/embedding_lookup/Identity:output:0*
T0*"
_output_shapes
:d
2*
(embedding_11/embedding_lookup/Identity_1�
simple_rnn_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_7/transpose/perm�
simple_rnn_7/transpose	Transpose1embedding_11/embedding_lookup/Identity_1:output:0$simple_rnn_7/transpose/perm:output:0*
T0*"
_output_shapes
:d
2
simple_rnn_7/transpose}
simple_rnn_7/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d      
   2
simple_rnn_7/Shape�
 simple_rnn_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_7/strided_slice/stack�
"simple_rnn_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_7/strided_slice/stack_1�
"simple_rnn_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_7/strided_slice/stack_2�
simple_rnn_7/strided_sliceStridedSlicesimple_rnn_7/Shape:output:0)simple_rnn_7/strided_slice/stack:output:0+simple_rnn_7/strided_slice/stack_1:output:0+simple_rnn_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_7/strided_slice�
(simple_rnn_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(simple_rnn_7/TensorArrayV2/element_shape�
simple_rnn_7/TensorArrayV2TensorListReserve1simple_rnn_7/TensorArrayV2/element_shape:output:0#simple_rnn_7/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_7/TensorArrayV2�
Bsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   2D
Bsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape�
4simple_rnn_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_7/transpose:y:0Ksimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_7/TensorArrayUnstack/TensorListFromTensor�
"simple_rnn_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_7/strided_slice_1/stack�
$simple_rnn_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_7/strided_slice_1/stack_1�
$simple_rnn_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_7/strided_slice_1/stack_2�
simple_rnn_7/strided_slice_1StridedSlicesimple_rnn_7/transpose:y:0+simple_rnn_7/strided_slice_1/stack:output:0-simple_rnn_7/strided_slice_1/stack_1:output:0-simple_rnn_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
*
shrink_axis_mask2
simple_rnn_7/strided_slice_1�
4simple_rnn_7/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp=simple_rnn_7_simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

:
_*
dtype026
4simple_rnn_7/simple_rnn_cell_8/MatMul/ReadVariableOp�
%simple_rnn_7/simple_rnn_cell_8/MatMulMatMul%simple_rnn_7/strided_slice_1:output:0<simple_rnn_7/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2'
%simple_rnn_7/simple_rnn_cell_8/MatMul�
5simple_rnn_7/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_7_simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype027
5simple_rnn_7/simple_rnn_cell_8/BiasAdd/ReadVariableOp�
&simple_rnn_7/simple_rnn_cell_8/BiasAddBiasAdd/simple_rnn_7/simple_rnn_cell_8/MatMul:product:0=simple_rnn_7/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2(
&simple_rnn_7/simple_rnn_cell_8/BiasAdd�
6simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype028
6simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp�
8simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp_1ReadVariableOpAsimple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02:
8simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp_1�
'simple_rnn_7/simple_rnn_cell_8/MatMul_1MatMul>simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0@simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:_2)
'simple_rnn_7/simple_rnn_cell_8/MatMul_1�
"simple_rnn_7/simple_rnn_cell_8/addAddV2/simple_rnn_7/simple_rnn_cell_8/BiasAdd:output:01simple_rnn_7/simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2$
"simple_rnn_7/simple_rnn_cell_8/add�
#simple_rnn_7/simple_rnn_cell_8/TanhTanh&simple_rnn_7/simple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2%
#simple_rnn_7/simple_rnn_cell_8/Tanh�
*simple_rnn_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2,
*simple_rnn_7/TensorArrayV2_1/element_shape�
simple_rnn_7/TensorArrayV2_1TensorListReserve3simple_rnn_7/TensorArrayV2_1/element_shape:output:0#simple_rnn_7/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_7/TensorArrayV2_1h
simple_rnn_7/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_7/time�
simple_rnn_7/ReadVariableOpReadVariableOp?simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02
simple_rnn_7/ReadVariableOp�
%simple_rnn_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%simple_rnn_7/while/maximum_iterations�
simple_rnn_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_7/while/loop_counter�
simple_rnn_7/whileWhile(simple_rnn_7/while/loop_counter:output:0.simple_rnn_7/while/maximum_iterations:output:0simple_rnn_7/time:output:0%simple_rnn_7/TensorArrayV2_1:handle:0#simple_rnn_7/ReadVariableOp:value:0#simple_rnn_7/strided_slice:output:0Dsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_7_simple_rnn_cell_8_matmul_readvariableop_resource>simple_rnn_7_simple_rnn_cell_8_biasadd_readvariableop_resourceAsimple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :_: : : : : *%
_read_only_resource_inputs
	*)
body!R
simple_rnn_7_while_body_60317*)
cond!R
simple_rnn_7_while_cond_60316*/
output_shapes
: : : : :_: : : : : *
parallel_iterations 2
simple_rnn_7/while�
=simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2?
=simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape�
/simple_rnn_7/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_7/while:output:3Fsimple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:d_*
element_dtype021
/simple_rnn_7/TensorArrayV2Stack/TensorListStack�
"simple_rnn_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2$
"simple_rnn_7/strided_slice_2/stack�
$simple_rnn_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_7/strided_slice_2/stack_1�
$simple_rnn_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_7/strided_slice_2/stack_2�
simple_rnn_7/strided_slice_2StridedSlice8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_7/strided_slice_2/stack:output:0-simple_rnn_7/strided_slice_2/stack_1:output:0-simple_rnn_7/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:_*
shrink_axis_mask2
simple_rnn_7/strided_slice_2�
simple_rnn_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_7/transpose_1/perm�
simple_rnn_7/transpose_1	Transpose8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_7/transpose_1/perm:output:0*
T0*"
_output_shapes
:d_2
simple_rnn_7/transpose_1�
simple_rnn_7/AssignVariableOpAssignVariableOp?simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resourcesimple_rnn_7/while:output:4^simple_rnn_7/ReadVariableOp7^simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
_output_shapes
 *
dtype02
simple_rnn_7/AssignVariableOp�
IdentityIdentitysimple_rnn_7/transpose_1:y:0^simple_rnn_7/AssignVariableOp^simple_rnn_7/while*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*5
_input_shapes$
":d:::::2>
simple_rnn_7/AssignVariableOpsimple_rnn_7/AssignVariableOp2(
simple_rnn_7/whilesimple_rnn_7/while:U Q
+
_output_shapes
:���������d
"
_user_specified_name
input_12
�
�
G__inference_embedding_11_layer_call_and_return_conditional_losses_59921

inputs
embedding_lookup_59915
identity�T
CastCastinputs*

DstT0*

SrcT0*
_output_shapes

:d2
Cast�
embedding_lookupResourceGatherembedding_lookup_59915Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/59915*"
_output_shapes
:d
*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/59915*"
_output_shapes
:d
2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*"
_output_shapes
:d
2
embedding_lookup/Identity_1s
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*"
_output_shapes
:d
2

Identity"
identityIdentity:output:0*!
_input_shapes
:d::F B

_output_shapes

:d
 
_user_specified_nameinputs
�
�
+sequential_11_simple_rnn_7_while_cond_59271R
Nsequential_11_simple_rnn_7_while_sequential_11_simple_rnn_7_while_loop_counterX
Tsequential_11_simple_rnn_7_while_sequential_11_simple_rnn_7_while_maximum_iterations0
,sequential_11_simple_rnn_7_while_placeholder2
.sequential_11_simple_rnn_7_while_placeholder_12
.sequential_11_simple_rnn_7_while_placeholder_2R
Nsequential_11_simple_rnn_7_while_less_sequential_11_simple_rnn_7_strided_slicei
esequential_11_simple_rnn_7_while_sequential_11_simple_rnn_7_while_cond_59271___redundant_placeholder0i
esequential_11_simple_rnn_7_while_sequential_11_simple_rnn_7_while_cond_59271___redundant_placeholder1i
esequential_11_simple_rnn_7_while_sequential_11_simple_rnn_7_while_cond_59271___redundant_placeholder2i
esequential_11_simple_rnn_7_while_sequential_11_simple_rnn_7_while_cond_59271___redundant_placeholder3-
)sequential_11_simple_rnn_7_while_identity
�
%sequential_11/simple_rnn_7/while/LessLess,sequential_11_simple_rnn_7_while_placeholderNsequential_11_simple_rnn_7_while_less_sequential_11_simple_rnn_7_strided_slice*
T0*
_output_shapes
: 2'
%sequential_11/simple_rnn_7/while/Less�
)sequential_11/simple_rnn_7/while/IdentityIdentity)sequential_11/simple_rnn_7/while/Less:z:0*
T0
*
_output_shapes
: 2+
)sequential_11/simple_rnn_7/while/Identity"_
)sequential_11_simple_rnn_7_while_identity2sequential_11/simple_rnn_7/while/Identity:output:0*7
_input_shapes&
$: : : : :_: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_59967
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_59967___redundant_placeholder03
/while_while_cond_59967___redundant_placeholder13
/while_while_cond_59967___redundant_placeholder23
/while_while_cond_59967___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*7
_input_shapes&
$: : : : :_: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
:
�
�
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_59495

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
_*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:_*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2	
BiasAdd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:__*
dtype02
MatMul_1/ReadVariableOpp
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:_2

MatMul_1b
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*
_output_shapes

:_2
addF
TanhTanhadd:z:0*
T0*
_output_shapes

:_2
TanhS
IdentityIdentityTanh:y:0*
T0*
_output_shapes

:_2

IdentityW

Identity_1IdentityTanh:y:0*
T0*
_output_shapes

:_2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*3
_input_shapes"
 :
:_::::F B

_output_shapes

:

 
_user_specified_nameinputs:FB

_output_shapes

:_
 
_user_specified_namestates
�<
�
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_61149

inputs4
0simple_rnn_cell_8_matmul_readvariableop_resource5
1simple_rnn_cell_8_biasadd_readvariableop_resource6
2simple_rnn_cell_8_matmul_1_readvariableop_resource8
4simple_rnn_cell_8_matmul_1_readvariableop_1_resource
identity��AssignVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permq
	transpose	Transposeinputstranspose/perm:output:0*
T0*"
_output_shapes
:d
2
	transposec
ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d      
   2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
*
shrink_axis_mask2
strided_slice_1�
'simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

:
_*
dtype02)
'simple_rnn_cell_8/MatMul/ReadVariableOp�
simple_rnn_cell_8/MatMulMatMulstrided_slice_1:output:0/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/MatMul�
(simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype02*
(simple_rnn_cell_8/BiasAdd/ReadVariableOp�
simple_rnn_cell_8/BiasAddBiasAdd"simple_rnn_cell_8/MatMul:product:00simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/BiasAdd�
)simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02+
)simple_rnn_cell_8/MatMul_1/ReadVariableOp�
+simple_rnn_cell_8/MatMul_1/ReadVariableOp_1ReadVariableOp4simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02-
+simple_rnn_cell_8/MatMul_1/ReadVariableOp_1�
simple_rnn_cell_8/MatMul_1MatMul1simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:03simple_rnn_cell_8/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/MatMul_1�
simple_rnn_cell_8/addAddV2"simple_rnn_cell_8/BiasAdd:output:0$simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/add|
simple_rnn_cell_8/TanhTanhsimple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2
simple_rnn_cell_8/Tanh�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time�
ReadVariableOpReadVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02
ReadVariableOp
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_8_matmul_readvariableop_resource1simple_rnn_cell_8_biasadd_readvariableop_resource4simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :_: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_61083*
condR
while_cond_61082*/
output_shapes
: : : : :_: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:d_*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:_*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*"
_output_shapes
:d_2
transpose_1�
AssignVariableOpAssignVariableOp2simple_rnn_cell_8_matmul_1_readvariableop_resourcewhile:output:4^ReadVariableOp*^simple_rnn_cell_8/MatMul_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpy
IdentityIdentitytranspose_1:y:0^AssignVariableOp^while*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*1
_input_shapes 
:d
::::2$
AssignVariableOpAssignVariableOp2
whilewhile:J F
"
_output_shapes
:d

 
_user_specified_nameinputs
�
�
while_cond_61184
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_61184___redundant_placeholder03
/while_while_cond_61184___redundant_placeholder13
/while_while_cond_61184___redundant_placeholder23
/while_while_cond_61184___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*7
_input_shapes&
$: : : : :_: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
:
�3
�
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_59760

inputs
simple_rnn_cell_8_59682
simple_rnn_cell_8_59684
simple_rnn_cell_8_59686
simple_rnn_cell_8_59688
identity��AssignVariableOp�)simple_rnn_cell_8/StatefulPartitionedCall�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������
2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
*
shrink_axis_mask2
strided_slice_1�
)simple_rnn_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0simple_rnn_cell_8_59682simple_rnn_cell_8_59684simple_rnn_cell_8_59686simple_rnn_cell_8_59688*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:_:_*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_593752+
)simple_rnn_cell_8/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
timex
ReadVariableOpReadVariableOpsimple_rnn_cell_8_59682*
_output_shapes

:_*
dtype02
ReadVariableOp
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_8_59684simple_rnn_cell_8_59686simple_rnn_cell_8_59688*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :_: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_59697*
condR
while_cond_59696*/
output_shapes
: : : : :_: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������_*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:_*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������_2
transpose_1�
AssignVariableOpAssignVariableOpsimple_rnn_cell_8_59682while:output:4^ReadVariableOp*^simple_rnn_cell_8/StatefulPartitionedCall*
_output_shapes
 *
dtype02
AssignVariableOp�
IdentityIdentitytranspose_1:y:0^AssignVariableOp*^simple_rnn_cell_8/StatefulPartitionedCall^while*
T0*+
_output_shapes
:���������_2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������
::::2$
AssignVariableOpAssignVariableOp2V
)simple_rnn_cell_8/StatefulPartitionedCall)simple_rnn_cell_8/StatefulPartitionedCall2
whilewhile:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_61337

inputs

states"
matmul_readvariableop_resource#
biasadd_readvariableop_resource&
"matmul_1_readvariableop_1_resource
identity

identity_1��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
_*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:_*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2	
BiasAdds
MatMul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1/ReadVariableOp_1ReadVariableOp"matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02
MatMul_1/ReadVariableOp_1�
MatMul_1BatchMatMulV2MatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:2

MatMul_1[
addAddV2BiasAdd:output:0MatMul_1:output:0*
T0*
_output_shapes
:2
add@
TanhTanhadd:z:0*
T0*
_output_shapes
:2
TanhM
IdentityIdentityTanh:y:0*
T0*
_output_shapes
:2

IdentityQ

Identity_1IdentityTanh:y:0*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*-
_input_shapes
:
:::::F B

_output_shapes

:

 
_user_specified_nameinputs:&"
 
_user_specified_namestates
�
�
,__inference_simple_rnn_7_layer_call_fn_61034
inputs_0
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������_*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_597602
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������_2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������
::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������

"
_user_specified_name
inputs/0
�
�
while_cond_60069
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_60069___redundant_placeholder03
/while_while_cond_60069___redundant_placeholder13
/while_while_cond_60069___redundant_placeholder23
/while_while_cond_60069___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*7
_input_shapes&
$: : : : :_: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
:
�U
�
H__inference_sequential_11_layer_call_and_return_conditional_losses_60635

inputs'
#embedding_11_embedding_lookup_60531A
=simple_rnn_7_simple_rnn_cell_8_matmul_readvariableop_resourceB
>simple_rnn_7_simple_rnn_cell_8_biasadd_readvariableop_resourceC
?simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resourceE
Asimple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_1_resource
identity��simple_rnn_7/AssignVariableOp�simple_rnn_7/whileg
lambda_11/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
lambda_11/add/yv
lambda_11/addAddV2inputslambda_11/add/y:output:0*
T0*"
_output_shapes
:d2
lambda_11/add�
lambda_11/SqueezeSqueezelambda_11/add:z:0*
T0*
_output_shapes

:d*
squeeze_dims

���������2
lambda_11/Squeeze�
embedding_11/CastCastlambda_11/Squeeze:output:0*

DstT0*

SrcT0*
_output_shapes

:d2
embedding_11/Cast�
embedding_11/embedding_lookupResourceGather#embedding_11_embedding_lookup_60531embedding_11/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_11/embedding_lookup/60531*"
_output_shapes
:d
*
dtype02
embedding_11/embedding_lookup�
&embedding_11/embedding_lookup/IdentityIdentity&embedding_11/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_11/embedding_lookup/60531*"
_output_shapes
:d
2(
&embedding_11/embedding_lookup/Identity�
(embedding_11/embedding_lookup/Identity_1Identity/embedding_11/embedding_lookup/Identity:output:0*
T0*"
_output_shapes
:d
2*
(embedding_11/embedding_lookup/Identity_1�
simple_rnn_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_7/transpose/perm�
simple_rnn_7/transpose	Transpose1embedding_11/embedding_lookup/Identity_1:output:0$simple_rnn_7/transpose/perm:output:0*
T0*"
_output_shapes
:d
2
simple_rnn_7/transpose}
simple_rnn_7/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d      
   2
simple_rnn_7/Shape�
 simple_rnn_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_7/strided_slice/stack�
"simple_rnn_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_7/strided_slice/stack_1�
"simple_rnn_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_7/strided_slice/stack_2�
simple_rnn_7/strided_sliceStridedSlicesimple_rnn_7/Shape:output:0)simple_rnn_7/strided_slice/stack:output:0+simple_rnn_7/strided_slice/stack_1:output:0+simple_rnn_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_7/strided_slice�
(simple_rnn_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(simple_rnn_7/TensorArrayV2/element_shape�
simple_rnn_7/TensorArrayV2TensorListReserve1simple_rnn_7/TensorArrayV2/element_shape:output:0#simple_rnn_7/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_7/TensorArrayV2�
Bsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   2D
Bsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape�
4simple_rnn_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_7/transpose:y:0Ksimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_7/TensorArrayUnstack/TensorListFromTensor�
"simple_rnn_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_7/strided_slice_1/stack�
$simple_rnn_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_7/strided_slice_1/stack_1�
$simple_rnn_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_7/strided_slice_1/stack_2�
simple_rnn_7/strided_slice_1StridedSlicesimple_rnn_7/transpose:y:0+simple_rnn_7/strided_slice_1/stack:output:0-simple_rnn_7/strided_slice_1/stack_1:output:0-simple_rnn_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
*
shrink_axis_mask2
simple_rnn_7/strided_slice_1�
4simple_rnn_7/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp=simple_rnn_7_simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

:
_*
dtype026
4simple_rnn_7/simple_rnn_cell_8/MatMul/ReadVariableOp�
%simple_rnn_7/simple_rnn_cell_8/MatMulMatMul%simple_rnn_7/strided_slice_1:output:0<simple_rnn_7/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2'
%simple_rnn_7/simple_rnn_cell_8/MatMul�
5simple_rnn_7/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_7_simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype027
5simple_rnn_7/simple_rnn_cell_8/BiasAdd/ReadVariableOp�
&simple_rnn_7/simple_rnn_cell_8/BiasAddBiasAdd/simple_rnn_7/simple_rnn_cell_8/MatMul:product:0=simple_rnn_7/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2(
&simple_rnn_7/simple_rnn_cell_8/BiasAdd�
6simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype028
6simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp�
8simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp_1ReadVariableOpAsimple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02:
8simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp_1�
'simple_rnn_7/simple_rnn_cell_8/MatMul_1MatMul>simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0@simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:_2)
'simple_rnn_7/simple_rnn_cell_8/MatMul_1�
"simple_rnn_7/simple_rnn_cell_8/addAddV2/simple_rnn_7/simple_rnn_cell_8/BiasAdd:output:01simple_rnn_7/simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2$
"simple_rnn_7/simple_rnn_cell_8/add�
#simple_rnn_7/simple_rnn_cell_8/TanhTanh&simple_rnn_7/simple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2%
#simple_rnn_7/simple_rnn_cell_8/Tanh�
*simple_rnn_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2,
*simple_rnn_7/TensorArrayV2_1/element_shape�
simple_rnn_7/TensorArrayV2_1TensorListReserve3simple_rnn_7/TensorArrayV2_1/element_shape:output:0#simple_rnn_7/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_7/TensorArrayV2_1h
simple_rnn_7/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_7/time�
simple_rnn_7/ReadVariableOpReadVariableOp?simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02
simple_rnn_7/ReadVariableOp�
%simple_rnn_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%simple_rnn_7/while/maximum_iterations�
simple_rnn_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_7/while/loop_counter�
simple_rnn_7/whileWhile(simple_rnn_7/while/loop_counter:output:0.simple_rnn_7/while/maximum_iterations:output:0simple_rnn_7/time:output:0%simple_rnn_7/TensorArrayV2_1:handle:0#simple_rnn_7/ReadVariableOp:value:0#simple_rnn_7/strided_slice:output:0Dsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_7_simple_rnn_cell_8_matmul_readvariableop_resource>simple_rnn_7_simple_rnn_cell_8_biasadd_readvariableop_resourceAsimple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :_: : : : : *%
_read_only_resource_inputs
	*)
body!R
simple_rnn_7_while_body_60569*)
cond!R
simple_rnn_7_while_cond_60568*/
output_shapes
: : : : :_: : : : : *
parallel_iterations 2
simple_rnn_7/while�
=simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2?
=simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape�
/simple_rnn_7/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_7/while:output:3Fsimple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:d_*
element_dtype021
/simple_rnn_7/TensorArrayV2Stack/TensorListStack�
"simple_rnn_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2$
"simple_rnn_7/strided_slice_2/stack�
$simple_rnn_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_7/strided_slice_2/stack_1�
$simple_rnn_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_7/strided_slice_2/stack_2�
simple_rnn_7/strided_slice_2StridedSlice8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_7/strided_slice_2/stack:output:0-simple_rnn_7/strided_slice_2/stack_1:output:0-simple_rnn_7/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:_*
shrink_axis_mask2
simple_rnn_7/strided_slice_2�
simple_rnn_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_7/transpose_1/perm�
simple_rnn_7/transpose_1	Transpose8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_7/transpose_1/perm:output:0*
T0*"
_output_shapes
:d_2
simple_rnn_7/transpose_1�
simple_rnn_7/AssignVariableOpAssignVariableOp?simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resourcesimple_rnn_7/while:output:4^simple_rnn_7/ReadVariableOp7^simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
_output_shapes
 *
dtype02
simple_rnn_7/AssignVariableOp�
IdentityIdentitysimple_rnn_7/transpose_1:y:0^simple_rnn_7/AssignVariableOp^simple_rnn_7/while*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*5
_input_shapes$
":d:::::2>
simple_rnn_7/AssignVariableOpsimple_rnn_7/AssignVariableOp2(
simple_rnn_7/whilesimple_rnn_7/while:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�U
�
H__inference_sequential_11_layer_call_and_return_conditional_losses_60494
input_12'
#embedding_11_embedding_lookup_60390A
=simple_rnn_7_simple_rnn_cell_8_matmul_readvariableop_resourceB
>simple_rnn_7_simple_rnn_cell_8_biasadd_readvariableop_resourceC
?simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resourceE
Asimple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_1_resource
identity��simple_rnn_7/AssignVariableOp�simple_rnn_7/whileg
lambda_11/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
lambda_11/add/yx
lambda_11/addAddV2input_12lambda_11/add/y:output:0*
T0*"
_output_shapes
:d2
lambda_11/add�
lambda_11/SqueezeSqueezelambda_11/add:z:0*
T0*
_output_shapes

:d*
squeeze_dims

���������2
lambda_11/Squeeze�
embedding_11/CastCastlambda_11/Squeeze:output:0*

DstT0*

SrcT0*
_output_shapes

:d2
embedding_11/Cast�
embedding_11/embedding_lookupResourceGather#embedding_11_embedding_lookup_60390embedding_11/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_11/embedding_lookup/60390*"
_output_shapes
:d
*
dtype02
embedding_11/embedding_lookup�
&embedding_11/embedding_lookup/IdentityIdentity&embedding_11/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_11/embedding_lookup/60390*"
_output_shapes
:d
2(
&embedding_11/embedding_lookup/Identity�
(embedding_11/embedding_lookup/Identity_1Identity/embedding_11/embedding_lookup/Identity:output:0*
T0*"
_output_shapes
:d
2*
(embedding_11/embedding_lookup/Identity_1�
simple_rnn_7/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_7/transpose/perm�
simple_rnn_7/transpose	Transpose1embedding_11/embedding_lookup/Identity_1:output:0$simple_rnn_7/transpose/perm:output:0*
T0*"
_output_shapes
:d
2
simple_rnn_7/transpose}
simple_rnn_7/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"d      
   2
simple_rnn_7/Shape�
 simple_rnn_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 simple_rnn_7/strided_slice/stack�
"simple_rnn_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_7/strided_slice/stack_1�
"simple_rnn_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"simple_rnn_7/strided_slice/stack_2�
simple_rnn_7/strided_sliceStridedSlicesimple_rnn_7/Shape:output:0)simple_rnn_7/strided_slice/stack:output:0+simple_rnn_7/strided_slice/stack_1:output:0+simple_rnn_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
simple_rnn_7/strided_slice�
(simple_rnn_7/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(simple_rnn_7/TensorArrayV2/element_shape�
simple_rnn_7/TensorArrayV2TensorListReserve1simple_rnn_7/TensorArrayV2/element_shape:output:0#simple_rnn_7/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_7/TensorArrayV2�
Bsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   2D
Bsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape�
4simple_rnn_7/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_7/transpose:y:0Ksimple_rnn_7/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4simple_rnn_7/TensorArrayUnstack/TensorListFromTensor�
"simple_rnn_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"simple_rnn_7/strided_slice_1/stack�
$simple_rnn_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_7/strided_slice_1/stack_1�
$simple_rnn_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_7/strided_slice_1/stack_2�
simple_rnn_7/strided_slice_1StridedSlicesimple_rnn_7/transpose:y:0+simple_rnn_7/strided_slice_1/stack:output:0-simple_rnn_7/strided_slice_1/stack_1:output:0-simple_rnn_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:
*
shrink_axis_mask2
simple_rnn_7/strided_slice_1�
4simple_rnn_7/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp=simple_rnn_7_simple_rnn_cell_8_matmul_readvariableop_resource*
_output_shapes

:
_*
dtype026
4simple_rnn_7/simple_rnn_cell_8/MatMul/ReadVariableOp�
%simple_rnn_7/simple_rnn_cell_8/MatMulMatMul%simple_rnn_7/strided_slice_1:output:0<simple_rnn_7/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2'
%simple_rnn_7/simple_rnn_cell_8/MatMul�
5simple_rnn_7/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_7_simple_rnn_cell_8_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype027
5simple_rnn_7/simple_rnn_cell_8/BiasAdd/ReadVariableOp�
&simple_rnn_7/simple_rnn_cell_8/BiasAddBiasAdd/simple_rnn_7/simple_rnn_cell_8/MatMul:product:0=simple_rnn_7/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2(
&simple_rnn_7/simple_rnn_cell_8/BiasAdd�
6simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype028
6simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp�
8simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp_1ReadVariableOpAsimple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
_output_shapes

:__*
dtype02:
8simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp_1�
'simple_rnn_7/simple_rnn_cell_8/MatMul_1MatMul>simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0@simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:_2)
'simple_rnn_7/simple_rnn_cell_8/MatMul_1�
"simple_rnn_7/simple_rnn_cell_8/addAddV2/simple_rnn_7/simple_rnn_cell_8/BiasAdd:output:01simple_rnn_7/simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2$
"simple_rnn_7/simple_rnn_cell_8/add�
#simple_rnn_7/simple_rnn_cell_8/TanhTanh&simple_rnn_7/simple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2%
#simple_rnn_7/simple_rnn_cell_8/Tanh�
*simple_rnn_7/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2,
*simple_rnn_7/TensorArrayV2_1/element_shape�
simple_rnn_7/TensorArrayV2_1TensorListReserve3simple_rnn_7/TensorArrayV2_1/element_shape:output:0#simple_rnn_7/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
simple_rnn_7/TensorArrayV2_1h
simple_rnn_7/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
simple_rnn_7/time�
simple_rnn_7/ReadVariableOpReadVariableOp?simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resource*
_output_shapes

:_*
dtype02
simple_rnn_7/ReadVariableOp�
%simple_rnn_7/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%simple_rnn_7/while/maximum_iterations�
simple_rnn_7/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
simple_rnn_7/while/loop_counter�
simple_rnn_7/whileWhile(simple_rnn_7/while/loop_counter:output:0.simple_rnn_7/while/maximum_iterations:output:0simple_rnn_7/time:output:0%simple_rnn_7/TensorArrayV2_1:handle:0#simple_rnn_7/ReadVariableOp:value:0#simple_rnn_7/strided_slice:output:0Dsimple_rnn_7/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_7_simple_rnn_cell_8_matmul_readvariableop_resource>simple_rnn_7_simple_rnn_cell_8_biasadd_readvariableop_resourceAsimple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_1_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*0
_output_shapes
: : : : :_: : : : : *%
_read_only_resource_inputs
	*)
body!R
simple_rnn_7_while_body_60428*)
cond!R
simple_rnn_7_while_cond_60427*/
output_shapes
: : : : :_: : : : : *
parallel_iterations 2
simple_rnn_7/while�
=simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   _   2?
=simple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape�
/simple_rnn_7/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_7/while:output:3Fsimple_rnn_7/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:d_*
element_dtype021
/simple_rnn_7/TensorArrayV2Stack/TensorListStack�
"simple_rnn_7/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2$
"simple_rnn_7/strided_slice_2/stack�
$simple_rnn_7/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$simple_rnn_7/strided_slice_2/stack_1�
$simple_rnn_7/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$simple_rnn_7/strided_slice_2/stack_2�
simple_rnn_7/strided_slice_2StridedSlice8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_7/strided_slice_2/stack:output:0-simple_rnn_7/strided_slice_2/stack_1:output:0-simple_rnn_7/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:_*
shrink_axis_mask2
simple_rnn_7/strided_slice_2�
simple_rnn_7/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
simple_rnn_7/transpose_1/perm�
simple_rnn_7/transpose_1	Transpose8simple_rnn_7/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_7/transpose_1/perm:output:0*
T0*"
_output_shapes
:d_2
simple_rnn_7/transpose_1�
simple_rnn_7/AssignVariableOpAssignVariableOp?simple_rnn_7_simple_rnn_cell_8_matmul_1_readvariableop_resourcesimple_rnn_7/while:output:4^simple_rnn_7/ReadVariableOp7^simple_rnn_7/simple_rnn_cell_8/MatMul_1/ReadVariableOp*
_output_shapes
 *
dtype02
simple_rnn_7/AssignVariableOp�
IdentityIdentitysimple_rnn_7/transpose_1:y:0^simple_rnn_7/AssignVariableOp^simple_rnn_7/while*
T0*"
_output_shapes
:d_2

Identity"
identityIdentity:output:0*5
_input_shapes$
":d:::::2>
simple_rnn_7/AssignVariableOpsimple_rnn_7/AssignVariableOp2(
simple_rnn_7/whilesimple_rnn_7/while:U Q
+
_output_shapes
:���������d
"
_user_specified_name
input_12
�)
�
while_body_60853
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0<
8while_simple_rnn_cell_8_matmul_readvariableop_resource_0=
9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0>
:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor:
6while_simple_rnn_cell_8_matmul_readvariableop_resource;
7while_simple_rnn_cell_8_biasadd_readvariableop_resource<
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource��
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:
*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
-while/simple_rnn_cell_8/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_8_matmul_readvariableop_resource_0*
_output_shapes

:
_*
dtype02/
-while/simple_rnn_cell_8/MatMul/ReadVariableOp�
while/simple_rnn_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:_2 
while/simple_rnn_cell_8/MatMul�
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0*
_output_shapes
:_*
dtype020
.while/simple_rnn_cell_8/BiasAdd/ReadVariableOp�
while/simple_rnn_cell_8/BiasAddBiasAdd(while/simple_rnn_cell_8/MatMul:product:06while/simple_rnn_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_2!
while/simple_rnn_cell_8/BiasAdd�
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes

:__*
dtype021
/while/simple_rnn_cell_8/MatMul_1/ReadVariableOp�
 while/simple_rnn_cell_8/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:_2"
 while/simple_rnn_cell_8/MatMul_1�
while/simple_rnn_cell_8/addAddV2(while/simple_rnn_cell_8/BiasAdd:output:0*while/simple_rnn_cell_8/MatMul_1:product:0*
T0*
_output_shapes

:_2
while/simple_rnn_cell_8/add�
while/simple_rnn_cell_8/TanhTanhwhile/simple_rnn_cell_8/add:z:0*
T0*
_output_shapes

:_2
while/simple_rnn_cell_8/Tanh�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_8/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3{
while/Identity_4Identity while/simple_rnn_cell_8/Tanh:y:0*
T0*
_output_shapes

:_2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_8_biasadd_readvariableop_resource9while_simple_rnn_cell_8_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_8_matmul_1_readvariableop_resource:while_simple_rnn_cell_8_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_8_matmul_readvariableop_resource8while_simple_rnn_cell_8_matmul_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*5
_input_shapes$
": : : : :_: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:_:

_output_shapes
: :

_output_shapes
: 
�
�
1__inference_simple_rnn_cell_8_layer_call_fn_61381

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_613702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*-
_input_shapes
:
::::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:

 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
�
`
D__inference_lambda_11_layer_call_and_return_conditional_losses_60783

inputs
identityS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
add/yX
addAddV2inputsadd/y:output:0*
T0*"
_output_shapes
:d2
addo
SqueezeSqueezeadd:z:0*
T0*
_output_shapes

:d*
squeeze_dims

���������2	
Squeeze[
IdentityIdentitySqueeze:output:0*
T0*
_output_shapes

:d2

Identity"
identityIdentity:output:0*!
_input_shapes
:d:J F
"
_output_shapes
:d
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
8
input_12,
serving_default_input_12:0d;
simple_rnn_7+
StatefulPartitionedCall:0d_tensorflow/serving/predict:��
�$
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api

signatures
:_default_save_signature
;__call__
*<&call_and_return_all_conditional_losses"�"
_tf_keras_sequential�!{"class_name": "Sequential", "name": "sequential_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_11", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 100, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}}, {"class_name": "Lambda", "config": {"name": "lambda_11", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABDAAAAcxQAAAB0AGoBfABkARcAZAJnAWQDjQJTACkETukBAAAA6f//\n//8pAdoEYXhpcykC2gJ0ZtoHc3F1ZWV6ZSkB2gF4qQByBwAAAPofPGlweXRob24taW5wdXQtMTUt\nM2E0OTI5NGJiODRhPtoIPGxhbWJkYT4yAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "Embedding", "config": {"name": "embedding_11", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "input_dim": 96, "output_dim": 10, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 100}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_7", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": true, "unroll": false, "time_major": false, "units": 95, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 100, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_11", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 100, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_12"}}, {"class_name": "Lambda", "config": {"name": "lambda_11", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABDAAAAcxQAAAB0AGoBfABkARcAZAJnAWQDjQJTACkETukBAAAA6f//\n//8pAdoEYXhpcykC2gJ0ZtoHc3F1ZWV6ZSkB2gF4qQByBwAAAPofPGlweXRob24taW5wdXQtMTUt\nM2E0OTI5NGJiODRhPtoIPGxhbWJkYT4yAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "Embedding", "config": {"name": "embedding_11", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "input_dim": 96, "output_dim": 10, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 100}}, {"class_name": "SimpleRNN", "config": {"name": "simple_rnn_7", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": true, "unroll": false, "time_major": false, "units": 95, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}]}}}
�
		variables

trainable_variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Lambda", "name": "lambda_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_11", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAQAAABDAAAAcxQAAAB0AGoBfABkARcAZAJnAWQDjQJTACkETukBAAAA6f//\n//8pAdoEYXhpcykC2gJ0ZtoHc3F1ZWV6ZSkB2gF4qQByBwAAAPofPGlweXRob24taW5wdXQtMTUt\nM2E0OTI5NGJiODRhPtoIPGxhbWJkYT4yAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
�

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
*@&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_11", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "input_dim": 96, "output_dim": 10, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 100}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 100]}}
�

cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
A__call__
*B&call_and_return_all_conditional_losses"�	
_tf_keras_rnn_layer�	{"class_name": "SimpleRNN", "name": "simple_rnn_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": true, "must_restore_from_config": false, "config": {"name": "simple_rnn_7", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": true, "unroll": false, "time_major": false, "units": 95, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [1, null, 10]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [1, 100, 10]}}
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
layer_regularization_losses
	variables
layer_metrics
trainable_variables
non_trainable_variables

layers
metrics
regularization_losses
;__call__
:_default_save_signature
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
,
Cserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
 layer_regularization_losses
		variables
!layer_metrics

trainable_variables

"layers
#metrics
regularization_losses
$non_trainable_variables
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
):'`
2embedding_11/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
%layer_regularization_losses
	variables
&layer_metrics
trainable_variables

'layers
(metrics
regularization_losses
)non_trainable_variables
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
�

kernel
recurrent_kernel
bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
D__call__
*E&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_cell_8", "trainable": true, "dtype": "float32", "units": 95, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
.layer_regularization_losses
	variables
/layer_metrics

0states
trainable_variables
1non_trainable_variables

2layers
3metrics
regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
7:5
_2%simple_rnn_7/simple_rnn_cell_8/kernel
A:?__2/simple_rnn_7/simple_rnn_cell_8/recurrent_kernel
1:/_2#simple_rnn_7/simple_rnn_cell_8/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
4layer_regularization_losses
*	variables
5layer_metrics
+trainable_variables

6layers
7metrics
,regularization_losses
8non_trainable_variables
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
90"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
':%_2simple_rnn_7/Variable
�2�
 __inference__wrapped_model_59338�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *+�(
&�#
input_12���������d
�2�
-__inference_sequential_11_layer_call_fn_60509
-__inference_sequential_11_layer_call_fn_60761
-__inference_sequential_11_layer_call_fn_60776
-__inference_sequential_11_layer_call_fn_60524�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_sequential_11_layer_call_and_return_conditional_losses_60746
H__inference_sequential_11_layer_call_and_return_conditional_losses_60635
H__inference_sequential_11_layer_call_and_return_conditional_losses_60494
H__inference_sequential_11_layer_call_and_return_conditional_losses_60383�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_lambda_11_layer_call_fn_60795
)__inference_lambda_11_layer_call_fn_60800�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_lambda_11_layer_call_and_return_conditional_losses_60783
D__inference_lambda_11_layer_call_and_return_conditional_losses_60790�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_embedding_11_layer_call_fn_60817�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_embedding_11_layer_call_and_return_conditional_losses_60810�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_simple_rnn_7_layer_call_fn_61034
,__inference_simple_rnn_7_layer_call_fn_61047
,__inference_simple_rnn_7_layer_call_fn_61264
,__inference_simple_rnn_7_layer_call_fn_61277�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_60919
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_61149
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_61021
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_61251�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
3B1
#__inference_signature_wrapper_60272input_12
�2�
1__inference_simple_rnn_cell_8_layer_call_fn_61381
1__inference_simple_rnn_cell_8_layer_call_fn_61443
1__inference_simple_rnn_cell_8_layer_call_fn_61348
1__inference_simple_rnn_cell_8_layer_call_fn_61429�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_61398
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_61296
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_61315
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_61415�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 �
 __inference__wrapped_model_59338v95�2
+�(
&�#
input_12���������d
� "6�3
1
simple_rnn_7!�
simple_rnn_7d_�
G__inference_embedding_11_layer_call_and_return_conditional_losses_60810M&�#
�
�
inputsd
� " �
�
0d

� p
,__inference_embedding_11_layer_call_fn_60817@&�#
�
�
inputsd
� "�d
�
D__inference_lambda_11_layer_call_and_return_conditional_losses_60783R2�/
(�%
�
inputsd

 
p
� "�
�
0d
� �
D__inference_lambda_11_layer_call_and_return_conditional_losses_60790R2�/
(�%
�
inputsd

 
p 
� "�
�
0d
� r
)__inference_lambda_11_layer_call_fn_60795E2�/
(�%
�
inputsd

 
p
� "�dr
)__inference_lambda_11_layer_call_fn_60800E2�/
(�%
�
inputsd

 
p 
� "�d�
H__inference_sequential_11_layer_call_and_return_conditional_losses_60383h9=�:
3�0
&�#
input_12���������d
p

 
� " �
�
0d_
� �
H__inference_sequential_11_layer_call_and_return_conditional_losses_60494h9=�:
3�0
&�#
input_12���������d
p 

 
� " �
�
0d_
� �
H__inference_sequential_11_layer_call_and_return_conditional_losses_60635f9;�8
1�.
$�!
inputs���������d
p

 
� " �
�
0d_
� �
H__inference_sequential_11_layer_call_and_return_conditional_losses_60746f9;�8
1�.
$�!
inputs���������d
p 

 
� " �
�
0d_
� �
-__inference_sequential_11_layer_call_fn_60509[9=�:
3�0
&�#
input_12���������d
p

 
� "�d_�
-__inference_sequential_11_layer_call_fn_60524[9=�:
3�0
&�#
input_12���������d
p 

 
� "�d_�
-__inference_sequential_11_layer_call_fn_60761Y9;�8
1�.
$�!
inputs���������d
p

 
� "�d_�
-__inference_sequential_11_layer_call_fn_60776Y9;�8
1�.
$�!
inputs���������d
p 

 
� "�d_�
#__inference_signature_wrapper_60272y98�5
� 
.�+
)
input_12�
input_12d"6�3
1
simple_rnn_7!�
simple_rnn_7d_�
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_60919y9F�C
<�9
+�(
&�#
inputs/0���������


 
p

 
� ")�&
�
0���������_
� �
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_61021y9F�C
<�9
+�(
&�#
inputs/0���������


 
p 

 
� ")�&
�
0���������_
� �
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_61149`96�3
,�)
�
inputsd


 
p

 
� " �
�
0d_
� �
G__inference_simple_rnn_7_layer_call_and_return_conditional_losses_61251`96�3
,�)
�
inputsd


 
p 

 
� " �
�
0d_
� �
,__inference_simple_rnn_7_layer_call_fn_61034l9F�C
<�9
+�(
&�#
inputs/0���������


 
p

 
� "����������_�
,__inference_simple_rnn_7_layer_call_fn_61047l9F�C
<�9
+�(
&�#
inputs/0���������


 
p 

 
� "����������_�
,__inference_simple_rnn_7_layer_call_fn_61264S96�3
,�)
�
inputsd


 
p

 
� "�d_�
,__inference_simple_rnn_7_layer_call_fn_61277S96�3
,�)
�
inputsd


 
p 

 
� "�d_�
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_61296�g�d
]�Z
�
inputs

;�8
6�3	!�
�_
�

jstates/0VariableSpec
p
� "4�1
*�'
�
0/0
�
�
0/1/0
� �
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_61315�g�d
]�Z
�
inputs

;�8
6�3	!�
�_
�

jstates/0VariableSpec
p 
� "4�1
*�'
�
0/0
�
�
0/1/0
� �
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_61398�J�G
@�=
�
inputs

�
�
states/0_
p
� "@�=
6�3
�
0/0_
�
�
0/1/0_
� �
L__inference_simple_rnn_cell_8_layer_call_and_return_conditional_losses_61415�J�G
@�=
�
inputs

�
�
states/0_
p 
� "@�=
6�3
�
0/0_
�
�
0/1/0_
� �
1__inference_simple_rnn_cell_8_layer_call_fn_61348�g�d
]�Z
�
inputs

;�8
6�3	!�
�_
�

jstates/0VariableSpec
p
� "&�#
�	
0
�
�
1/0�
1__inference_simple_rnn_cell_8_layer_call_fn_61381�g�d
]�Z
�
inputs

;�8
6�3	!�
�_
�

jstates/0VariableSpec
p 
� "&�#
�	
0
�
�
1/0�
1__inference_simple_rnn_cell_8_layer_call_fn_61429�J�G
@�=
�
inputs

�
�
states/0_
p
� "2�/
�
0_
�
�
1/0_�
1__inference_simple_rnn_cell_8_layer_call_fn_61443�J�G
@�=
�
inputs

�
�
states/0_
p 
� "2�/
�
0_
�
�
1/0_