from dataclasses import dataclass
import typing
from typing import Iterable
import numpy as np

@dataclass
class _DefaultConfig:
  # algo to run
  algos: Iterable[str] = ('RBF',
                          'KES',
                          'KESFast',
                          'KESHiger',
                          'KESHigherFast',
                          'SES',
                          'SESFast')
  # data param
  dataset__name: Iteratble[str] = ('Particles','RoughVol')
  dataset__nb_bags: Iterable[int] = (50,)                
  dataset__nb_items: Iterable[int] = (50,)
  dataset__nb_time_steps: Iterable[int] = (200,)
  dataset__ymin = Iterable[float] = (1e0,)     
  dataset__ymax = Iterable[float] = (1e2,)    
  dataset__radius =  Iterable[float] = (3.5e-1,)  
  # algo param
  # TODO: hyperparameters should not be iterables 
  RBF__ : 
  KES__alphas :  Iterable[float] = ((1e-1,1e0,1e1,1e2),)  #TODO: KES should have rbf and others
  KESFast__alphas : Iterable[float] = ((1e-1,1e0,1e1,1e2),)
  KESFast__depths : Iterable[int] = (4,)
  KESFast__ncompos : Iterable[int] = (100,)
  KESFast__rbf : Iterable[bool] = (True,)
  KESHigherFast__depths1 : Iterable[int] = (4,)
  KESHigherFast__ncompos1 : Iterable[int] = (100,)
  KESHigherFast__rbf1 :  Iterable[bool] = (True,)
  KESHigherFast__alphas1 : Iterable[float] = ((1e-1,1e0,1e1,1e2),)
  KESHigherFast__lambdas_ : Iterable[float] = ((1e-1,1e0,1e1,1e2),)
  KESHigherFast__depths2 : Iterable[int] = (4,)
  KESHigherFast__ncompos2 : Iterable[int] = (100,)
  KESHigherFast__rbf2 : Iterable[bool] = (True,)
  KESHigherFast__alphas2 : Iterable[float] = ((1e-1,1e0,1e1,1e2),)
  SES__depths1 : Iterable[int] = (4,)
  SES__depths2 : Iterable[int] = (4,)
  SESFast__depths1 : Iterable[int] = (4,)
  SESFast__depths2 : Iterable[int] = (4,)
  SESFast__ncompos1 : Iterable[int] = (100,)
  SESFast__ncompos2 : Iterable[int] = (100,)
  SESFast__rbf : Iterable[bool] = (True,)
  SESFast__alpha : Iterable[float] = ((1e-1,1e0,1e1,1e2),)
  ll:  Iterable[int] = ((0),)
  at: Iterable[bool] = (True,)  
  num_trials: Iterable[int] = (5,)  
  cv:  Iterable[int] = (3,)  
                                           
@dataclass
class _DefaultConfigParticles(_DefaultConfig):
  dataset__name: Iteratble[str] = ('Particles')
  algos: Iterable[str] = ('RBF', 'KES','KESFast', 'KESHigher', 'KesHigherFast', 'SES', 'SESFast')
  dataset__ymin: Iterable[float] = (1e0,)
  dataset__ymax: Iterable[float] = (1e3,)

@dataclass
class _DefaultConfigRoughVol(_DefaultConfig):
  dataset__name: Iteratble[str] = ('RoughVol')
  algos: Iterable[str] = ('RBF', 'KES', 'KESFast', 'KesHigher','KesHigherFast', 'SES', 'SESFast')
  dataset__nb_bags: Iterable[int] = (50,)     
  dataset__nb_items: Iterable[int] = (25,50,75,100)
  dataset__nb_time_steps: Iterable[int] = (200,)
  dataset__ymin: Iterable[float] = (1e-6,)
  dataset__ymax: Iterable[float] = (1e0,)


@dataclass
class _VaryItems(_DefaultConfigRoughVol):
#   algos: Iterable[str] = ('RBF', 'KES', 'KESFast', 'KesHigher','KesHigherFast', 'SES', 'SESFast')
  algos: Iterable[str] = ('KESFast', 'SES')
  dataset__nb_items: Iterable[int] = (25,50,75,100)
  ll : Iterable[int] = ((0),)
  at: Iterable[bool] = (True,)  


# KESRoughVol = _VaryItems(
     
#  )

KESFastRoughVol = _VaryItems(
  algos: Iterable[str] = ('KESFast',)
  KESFast__alphas : Iterable[float] = ((5,10,15),)
  KESFast__depths : Iterable[int] = (4,)
  KESFast__ncompos : Iterable[int] = (100,)
  KESFast__rbf : Iterable[bool] = (True,)
 )

# KESHigerFastRoughVol = _VaryItems(
#   algos: Iterable[str] = ('KESHigherFast',)
#   KESHigherFast__depths1 :
#   KESHigherFast__ncompos1 : 
#   KESHigherFast__rbf1 :
#   KESHigherFast__alphas1 : 
#   KESHigherFast__lambdas_ :
#   KESHigherFast__depths2 : 
#   KESHigherFast__ncompos2 : 
#   KESHigherFast__rbf2 : 
#   KESHigherFast__alphas2 : 
#  )


# SESRoughVol = _VaryItems(
#   algos: Iterable[str] = ('SES',)
#   SES__depths1 : 
#   SES__depths2 :
#  )

# SESFastRoughVol = _VaryItems(
#   algos: Iterable[str] = ('SESFast',)
#   SESFast__depths1 : 
#   SESFast__depths2 :
#   SESFast__ncompos1 : 
#   SESFast__ncompos2 : 
#   SESFast__rbf : 
#   SESFast__alpha : 
#  )



