TITLE Implemented using kinetics from Booth et al 1997 by Avinash , SPINE LABS , IITH

NEURON {
	SUFFIX CaL
	USEION ca READ eca WRITE ica
	RANGE gcalbar       
    GLOBAL minf,taum
	THREADSAFE
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	
	gcalbar = 0.003 	(mho/cm2)
	eca  = 80	(mV)
}
STATE {
	m 
}
ASSIGNED {
	ica (mA/cm2)
    minf
    taum (ms)
	v (mV)
}

INITIAL {
         rates(v)
         m=minf
      
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	ica = gcalbar*m*(v - eca)
}

DERIVATIVE states {	
	  rates(v)
	m' = (minf - m)/taum
	
}


PROCEDURE rates(v(mV)) {  
	UNITSOFF	
		minf = 1/(1+ exp(-(v + 40)/7))
		taum = 40
	UNITSON	
}

