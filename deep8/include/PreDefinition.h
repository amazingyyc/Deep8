#ifndef DEEP8_PREDEFINITION_H
#define DEEP8_PREDEFINITION_H

namespace Deep8 {

/***********************************************************************************/
/**define float*/
/***********************************************************************************/
typedef Variable<float>          VariableF;
typedef Parameter<float>         ParameterF;
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
typedef ReLu<float>				 ReLuF;
typedef ReShape<float>			 ReShapeF;
typedef ScalarDivide<float>		 ScalarDivideF;
typedef ScalarMinus<float>		 ScalarMinusF;
typedef Sigmoid<float>			 SigmoidF;
typedef Softmax<float>			 SoftmaxF;
typedef Square<float>			 SquareF;
typedef Tanh<float>				 TanhF;

typedef SGDTrainer<float>		 SGDTrainerF;
typedef AdagradTrainer<float>	 AdagradTrainerF;
typedef AdamTrainer<float>		 AdamTrainerF;
typedef RMSPropTrainer<float>    RMSPropTrainerF;
typedef MomentumTrainer<float>	 MomentumTrainerF;
typedef Executor<float>			 ExecutorF;
typedef EagerExecutor<float>     EagerExecutorF;
typedef Expression<float>        ExpressionF;


/***********************************************************************************/
/**define double*/
/***********************************************************************************/
typedef Variable<double>          VariableD;
typedef Parameter<double>         ParameterD;
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
typedef ReLu<double>			  ReLuD;
typedef ReShape<double>			  ReShapeD;
typedef ScalarDivide<double>	  ScalarDivideD;
typedef ScalarMinus<double>		  ScalarMinusD;
typedef Sigmoid<double>			  SigmoidD;
typedef Softmax<double>			  SoftmaxD;
typedef Square<double>			  SquareD;
typedef Tanh<double>			  TanhD;

typedef SGDTrainer<double>		  SGDTrainerD;
typedef AdagradTrainer<double>	  AdagradTrainerD;
typedef AdamTrainer<double>		  AdamTrainerD;
typedef RMSPropTrainer<double>    RMSPropTrainerD;
typedef MomentumTrainer<double>	  MomentumTrainerD;
typedef Executor<double>          ExecutorD;
typedef EagerExecutor<double>	  EagerExecutorD;
typedef Expression<double>        ExpressionD;

/***********************************************************************************/
/**define half*/
/***********************************************************************************/
#ifdef HAVE_HALF
typedef Variable<half>          VariableH;
typedef Parameter<half>         ParameterH;
typedef Abs<half>				AbsH;
typedef Add<half>				AddH;
typedef AddScalar<half>		    AddScalarH;
typedef AvgPooling2d<half>	    AvgPooling2dH;
typedef Conv2d<half>			Conv2dH;
typedef DeConv2d<half>		    DeConv2dH;
typedef Divide<half>			DivideH;
typedef DivideScalar<half>	    DivideScalarH;
typedef Exp<half>				ExpH;
typedef L1Norm<half>			L1NormH;
typedef L2Norm<half>			L2NormH;
typedef Linear<half>			LinearH;
typedef Log<half>				LogH;
typedef LReLu<half>			    LReLuH;
typedef MatrixMultiply<half>	MatrixMultiplyH;
typedef MaxPooling2d<half>	    MaxPooling2dH;
typedef Minus<half>			    MinusH;
typedef MinusScalar<half>	    MinusScalarH;
typedef Multiply<half>		    MultiplyH;
typedef MultiplyScalar<half>	MultiplyScalarH;
typedef ReLu<half>			    ReLuH;
typedef ReShape<half>			ReShapeH;
typedef ScalarDivide<half>	    ScalarDivideH;
typedef ScalarMinus<half>		ScalarMinusH;
typedef Sigmoid<half>			SigmoidH;
typedef Softmax<half>	        SoftmaxH;
typedef Square<half>			SquareH;
typedef Tanh<half>			    TanhH;

typedef SGDTrainer<half>		SGDTrainerH;
typedef AdagradTrainer<half>	AdagradTrainerH;
typedef AdamTrainer<half>		AdamTrainerH;
typedef RMSPropTrainer<half>    RMSPropTrainerH;
typedef MomentumTrainer<half>	MomentumTrainerH;
typedef Executor<half>          ExecutorH;
typedef EagerExecutor<half>     EagerExecutorH;
typedef Expression<half>        ExpressionH;
#endif

}

#endif