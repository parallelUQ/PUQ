'''
This module defines the knobs of an intervention and forms
the available intervantions considerin school closures,
cocooning, and different levels od social distance.
'''
from numpy import exp, round, array
import itertools
from SEIYAHRD_sim import WEEKDAY, WEEKEND, HOLIDAY, LONG_HOLIDAY


class Intervension:
    def __init__(self, SC, CO, SD, epi, demographics):
        '''
            Attrs:
            school_closure (int): 0 schools are open, 1 schools are closed
            cocooning (float): level of cocooning [0,1)
            social_distance (float): level of social distance [0,1)
            epi (EpiParams): instance of the parameterization
            demographics (ndarray): Population demographics
        '''
        self.school_closure = SC
        self.cocooning = CO
        self.social_distance = SD
        self.cost = SC + CO + (round(exp(5 * SD), 3) - 1)
        demographics_normalized = demographics / demographics.sum()
        self.phi_weekday = epi.effective_phi(SC, CO, SD, demographics_normalized, day_type=WEEKDAY)
        self.phi_weekend = epi.effective_phi(SC, CO, SD, demographics_normalized, day_type=WEEKEND)
        self.phi_long_weekend = epi.effective_phi(SC, CO, SD, demographics_normalized, day_type=LONG_HOLIDAY)
    
    def phi(self, day_type):
        if day_type == WEEKDAY:
            return self.phi_weekday
        elif day_type == WEEKEND:
            return self.phi_weekend
        elif day_type == HOLIDAY:
            return self.phi_weekend
        elif day_type == LONG_HOLIDAY:
            return self.phi_long_weekend
        else:
            raise 'Day type not recognized'
    
    @property
    def SC(self):
        return self.school_closure
    
    @property
    def CO(self):
        return self.cocooning
    
    @property
    def SD(self):
        return self.social_distance


def create_intLevel(school_closure_levels, cocoon_levels, social_distance_levels):
    '''
        Obtain the intervention levels from discrete school closure, cocooning and social distance levels
    '''
    return [i for i in itertools.product(school_closure_levels, cocoon_levels, social_distance_levels)]


def form_interventions(intervention_levels, epi, demographics):
    '''
        Compile a intervention list from a given intervention levels, epiparams and demographics
    '''
    interventions = []
    for SC, CO, SD in intervention_levels:
        interventions.append(Intervension(SC, CO, SD, epi, demographics))
    return interventions
