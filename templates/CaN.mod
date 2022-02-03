TITLE Implemented using kinetics from Booth et al 1997 by Avinash , SPINE LABS , IITH

NEURON {
	SUFFIX CaN
	USEION ca READ eca WRITE ica
	RANGE gcanbar       
    GLOBAL hinf,minf,taum,tauh
	THREADSAFE
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gcanbar = 0.003 (mho/cm2)
	eca  = 80	(mV)
}
STATE {
	m h
}
ASSIGNED {
	ica (mA/cm2)
    minf
    hinf
    taum  (ms)
    tauh  (ms)
	v (mV)
}

INITIAL {
         rates(v)
         m=minf
         h=hinf
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	ica = gcanbar*m*m*h*(v - eca)
}

DERIVATIVE states {	
	  rates(v)
	m' = (minf - m)/taum
	h' = (hinf - h)/tauh
}


PROCEDURE rates(v(mV)) {  
		UNITSOFF
		minf = 1/(1+ exp(-(v + 30)/5))
		taum = 4
		hinf = 1/(1+exp((v + 45)/5))
		tauh = 40
		UNITSON
}

