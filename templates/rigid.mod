TITLE rigid
: Rigid body mechanics of ankle joint
: moment = -moment_of_inertia * theta''

NEURON {
	SUFFIX rigid
	GLOBAL moment
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
        moi=0.12 (kg m2)
        moment (kg m2 radian/s2)
}

STATE {
	theta avelo
}

INITIAL {
    theta = -30*3.1415/180.0
}

BREAKPOINT {
	SOLVE states METHOD cnexp
}

DERIVATIVE states {	
        UNITSOFF
        theta' = avelo
        avelo' = -moment/moi
        UNITSOFF
}

