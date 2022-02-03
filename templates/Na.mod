TITLE Implemented using kinetics from Booth et al 1997 by Avinash , SPINE LABS , IITH

NEURON {
	SUFFIX naf
	USEION na READ ena WRITE ina
	RANGE gnabar
	GLOBAL minf , mtau , hinf , htau
	THREADSAFE
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	
	gnabar = 0.120 	(mho/cm2)
	ena   = 55	(mV)
}
STATE {
	m h
}
ASSIGNED {
	ina (mA/cm2)
	minf 
    mtau (ms)
	hinf 
	htau (ms)
	v (mV)
}


INITIAL {
         rates(v)
	
         m=minf
         h=hinf
	
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	ina = gnabar*m*m*m*h*(v - ena)
}

DERIVATIVE states {	
	  rates(v)
	m' = (minf - m)/mtau
	h' = (hinf - h)/htau
}

UNITSOFF
PROCEDURE rates(v(mV)) {  
		
		minf = 1/(1+ exp(-(v + 35)/7.8))
		mtau = 1
		hinf = 1/(1+exp((v + 55)/7))
		htau = 30/((exp((v+50)/15) + exp(-(v+50)/16)))
		
}
UNITSON