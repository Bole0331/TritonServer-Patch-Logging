?

??
:
Add
x"T
y"T
z"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
.
Identity

input"T
output"T"	
Ttype
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
:
Sub
x"T
y"T
z"T"
Ttype:
2	"serve*
1.12.0-rc22
b'unknown'?
p
TENSOR_INPUT0Placeholder*
dtype0*'
_output_shapes
:?????????*
shape:?????????
p
TENSOR_INPUT1Placeholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
Z
ADDAddTENSOR_INPUT0TENSOR_INPUT1*
T0*'
_output_shapes
:?????????
Z
SUBSubTENSOR_INPUT0TENSOR_INPUT1*'
_output_shapes
:?????????*
T0
c
CAST0CastADD*

SrcT0*
Truncate( *

DstT0*'
_output_shapes
:?????????
c
CAST1CastSUB*

SrcT0*
Truncate( *

DstT0*'
_output_shapes
:?????????
S
TENSOR_OUTPUT0IdentityCAST0*'
_output_shapes
:?????????*
T0
S
TENSOR_OUTPUT1IdentityCAST1*
T0*'
_output_shapes
:????????? "*?
serving_default?
0
INPUT0&
TENSOR_INPUT0:0?????????
0
INPUT1&
TENSOR_INPUT1:0?????????2
OUTPUT0'
TENSOR_OUTPUT0:0?????????2
OUTPUT1'
TENSOR_OUTPUT1:0?????????tensorflow/serving/predict