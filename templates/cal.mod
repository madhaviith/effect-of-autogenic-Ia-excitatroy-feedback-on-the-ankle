:SOMA

: Marco Capogrosso & Emanuele Formento
:
:
: This model has been adapted and is described in detail in:
:
: McIntyre CC and Grill WM. Extracellular Stimulation of Central Neurons:
: Influence of Stimulus Waveform and Frequency on Neuronal Output
: Journal of Neurophysiology 88:1592-1604, 2002.

TITLE Motor Axon Soma
INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX caL
	
	NONSPECIFIC_CURRENT icaL
	RANGE   gcaL
	RANGE p_inf
	RANGE tau_p
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	
	gcaL = 0.0001  (mho/cm2)

	ca0 = 2  
	dt              (ms)
	v               (mV)
	
	R=8.314472
	F=96485.34
}

STATE {
	 p cai 
}

ASSIGNED {
	
	icaL  (mA/cm2)
	Eca  (mV)
	
	p_inf
	tau_p (ms)
	
}

BREAKPOINT {

	SOLVE states METHOD cnexp
	Eca = ((1000*R*309.15)/(2*F))*log(ca0/cai)*1(mV)
	icaL = gcaL*p*(v-Eca)
	
}


UNITSOFF
DERIVATIVE states {  
	 : exact Hodgkin-Huxley equations
        evaluate_fct(v)
	p' = (p_inf - p) / tau_p
	cai'= 0.01*(-(icaL) - 4*cai)
}



INITIAL {
	evaluate_fct(v)
	p = p_inf
	cai = 0.0001
}

PROCEDURE evaluate_fct(v(mV)) {
	
	    :L-type
	tau_p=40
	p_inf=1/(1+Exp(-(v+50)/3.7))

}


FUNCTION Exp(x) {
	if (x < -100) {
		Exp = 0
	}else{
		Exp = exp(x)
	}
}

UNITSON
