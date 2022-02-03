TITLE K-ca channel
:Implemented using kinetics from Booth et al 1997 by Avinash , SPINE LABS , IITH


NEURON {
	SUFFIX KCa
	USEION ca READ ica
	USEION k READ ek WRITE ik
    RANGE gkcabar 
    THREADSAFE	
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(mol)   = (1)
	(molar) = (1/liter)
	(mM)	= (millimolar)
}

PARAMETER {
	gkcabar = 0.003 (mho/cm2)
	frca = 0.01  :
	alp  = 0.009 : mol/C/um
	kca  = 2     : m/s
	kd   = 0.0002  (mM)
	ek  = 80 (mV)
}



STATE {
  cai(mM)
}

ASSIGNED { 
  ik (mA/cm2)
  ica  (mA/cm2)
  gkca 
  v(mV)
}

INITIAL { 
   cai = 0.0001 
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gkca = cai/(cai + kd)
	ik = gkcabar*gkca*(v-ek)

}

UNITSOFF
DERIVATIVE states {
	cai' = frca*(-alp*ica - kca*cai)
}

UNITSON

