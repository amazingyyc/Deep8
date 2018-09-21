#ifndef DEEP8_PREDEFINITION_H
#define DEEP8_PREDEFINITION_H

namespace Deep8 {

/**define type*/
typedef Variable<float>          VariableF;
typedef Parameter<float>         ParameterF;
typedef InputParameter<float>    InputParameterF;
typedef ConstantParameter<float> ConstantParameterF;
typedef Abs<float>				 AbsF;
typedef Add<float>				 AddF;
typedef AddScalar<float>		 AddScalarF;
typedef AvgPooling2d<float>		 AvgPooling2dF;
typedef Conv2d<float>			 Conv2dF;
typedef DeConv2d<float>			 DeConv2dF;
typedef Divide<float>			 DivideF;
typedef DivideScalar<float>		 DivideScalarF;
typedef Exp<float>				 ExpF;
typedef L1Norm<float>			 L1NormF;
typedef L2Norm<float>			 L2NormF;
typedef Linear<float>			 LinearF;
typedef Log<float>				 LogF;
typedef LReLu<float>			 LReLuF;
typedef MatrixMultiply<float>	 MatrixMultiplyF;
typedef MaxPooling2d<float>		 MaxPooling2dF;
typedef Minus<float>			 MinusF;
typedef MinusScalar<float>		 MinusScalarF;
typedef Multiply<float>			 MultiplyF;
typedef MultiplyScalar<float>	 MultiplyScalarF;
typedef Pow<float>				 PowF;
typedef ReLu<float>				 ReLuF;
typedef ReShape<float>			 ReShapeF;
typedef ScalarDivide<float>		 ScalarDivideF;
typedef ScalarMinus<float>		 ScalarMinusF;
typedef Sigmoid<float>			 SigmoidF;
typedef Softmax<float>			 SoftmaxF;
typedef Square<float>			 SquareF;
typedef SumElements<float>		 SumElementsF;
typedef TanH<float>				 TanHF;
typedef SGDTrainer<float>		 SGDTrainerF;
//typedef AdagradTrainer<float>	 AdagradTrainerF;
//typedef AdamTrainer<float>		 AdamTrainerF;
//typedef RMSPropTrainer<float>    RMSPropTrainerF;
//typedef MomentumTrainer<float>	 MomentumTrainerF;
typedef Executor<float>			 ExecutorF;
typedef DefaultExecutor<float>   DefaultExecutorF;
typedef Expression<float>        ExpressionF;


typedef Variable<double>          VariableD;
typedef Parameter<double>         ParameterD;
typedef InputParameter<double>    InputParameterD;
typedef ConstantParameter<double> ConstantParameterD;
typedef Abs<double>				  AbsD;
typedef Add<double>				  AddD;
typedef AddScalar<double>		  AddScalarD;
typedef AvgPooling2d<double>	  AvgPooling2dD;
typedef Conv2d<double>			  Conv2dD;
typedef DeConv2d<double>		  DeConv2dD;
typedef Divide<double>			  DivideD;
typedef DivideScalar<double>	  DivideScalarD;
typedef Exp<double>				  ExpD;
typedef L1Norm<double>			  L1NormD;
typedef L2Norm<double>			  L2NormD;
typedef Linear<double>			  LinearD;
typedef Log<double>				  LogD;
typedef LReLu<double>			  LReLuD;
typedef MatrixMultiply<double>	  MatrixMultiplyD;
typedef MaxPooling2d<double>	  MaxPooling2dD;
typedef Minus<double>			  MinusD;
typedef MinusScalar<double>		  MinusScalarD;
typedef Multiply<double>		  MultiplyD;
typedef MultiplyScalar<double>	  MultiplyScalarD;
typedef Pow<double>				  PowD;
typedef ReLu<double>			  ReLuD;
typedef ReShape<double>			  ReShapeD;
typedef ScalarDivide<double>	  ScalarDivideD;
typedef ScalarMinus<double>		  ScalarMinusD;
typedef Sigmoid<double>			  SigmoidD;
typedef Softmax<double>			  SoftmaxD;
typedef Square<double>			  SquareD;
typedef SumElements<double>		  SumElementsD;
typedef TanH<double>			  TanHD;
typedef SGDTrainer<double>		  SGDTrainerD;
//typedef AdagradTrainer<double>	  AdagradTrainerD;
//typedef AdamTrainer<double>		  AdamTrainerD;
//typedef RMSPropTrainer<double>    RMSPropTrainerD;
//typedef MomentumTrainer<double>	  MomentumTrainerD;
typedef Executor<double>          ExecutorD;
typedef DefaultExecutor<double>   DefaultExecutorD;
typedef Expression<double>        ExpressionD;

}

#endif