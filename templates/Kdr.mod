TITLE Implemented using kinetics from Booth et al 1997 by Avinash , SPINE LABS , IITH

NEURON {
	SUFFIX Kdr
	USEION k READ ek WRITE ik
	RANGE gkdrbar
	GLOBAL ninf , ntau 
	THREADSAFE
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	
	gkdrbar = 0.120 (mho/cm2)
	ek  = -80	(mV)
}
STATE {
	n 
}
ASSIGNED {
	ik (mA/cm2)
	ninf 
	ntau (ms)
	v (mV)
	
}

INITIAL {
    rates(v)
    n=ninf
	
}


BREAKPOINT {
	SOLVE states METHOD cnexp
	ik = gkdrbar*n*n*n*n*(v - ek)
}

DERIVATIVE states {	
	  rates(v)
	n' = (ninf - n)/ntau
	
}


PROCEDURE rates(v(mV)) {  
		UNITSOFF
		ninf = 1/(1+exp(-(v + 28)/15))
		ntau = 7/((exp((v+40)/40) + exp(-(v+40)/50)))
		UNITSON
}
