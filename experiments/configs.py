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
  dataset__name: Iterable[str] = ('Particles','RoughVol')
  dataset__nb_bags: Iterable[int] = (50,)                
  dataset__nb_items: Iterable[int] = (50,)
  dataset__nb_time_steps: Iterable[int] = (200,)
  dataset__ymin: Iterable[float] = (1e0,)     
  dataset__ymax: Iterable[float] = (1e2,)    
  dataset__radius:  Iterable[float] = (3.5e-1,)  
  # algo param
  # TODO: hyperparameters should not be iterables 
#   RBF__ : 
  KES__alphas :  Iterable[float] = ((1e-1,1e0,1e1,1e2),)  #TODO: KES should have rbf and others
  KES__rbf: Iterable[bool] = (True,)
  KES__dyadic_order: Iterable[int] = (1,)
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
  FDA__name: Iterable[str] = ('sqr','cov')
  FDA__alphas : Iterable[int] = (-1,)
  ll:  Iterable[int] = ((0,),)
  at: Iterable[bool] = (True,)  
  num_trials: Iterable[int] = (5,)  
  cv:  Iterable[int] = (3,)  
                                           

@dataclass
class _DefaultConfigRoughVol(_DefaultConfig):
  dataset__name: Iterable[str] = ('RoughVol',)
  algos: Iterable[str] = ('RBF', 'KES', 'KESFast', 'KesHigher','KesHigherFast', 'SES', 'SESFast')
  dataset__nb_bags: Iterable[int] = (100,)     
  dataset__nb_time_steps: Iterable[int] = (200,)
  dataset__ymin: Iterable[float] = (1e-6,)
  dataset__ymax: Iterable[float] = (1e0,)

@dataclass
class _DefaultConfigParticles(_DefaultConfig):
  dataset__name: Iterable[str] = ('Particles',)
  algos: Iterable[str] = ('RBF', 'KES', 'KESFast', 'KesHigher','KesHigherFast', 'SES', 'SESFast')
  dataset__nb_bags: Iterable[int] = (50,)
  dataset__nb_items: Iterable[int] = (20,)     
  dataset__nb_time_steps: Iterable[int] = (100,)
  dataset__ymin: Iterable[float] = (1e0,)
  dataset__ymax: Iterable[float] = (1e3,)


@dataclass
class _VaryItems(_DefaultConfigRoughVol):
#   algos: Iterable[str] = ('RBF', 'KES', 'KESFast', 'KesHigher','KesHigherFast', 'SES', 'SESFast')
  algos: Iterable[str] = ('KESFast', 'SES')
  dataset__nb_items: Iterable[int] = (100,)
  dataset__nb_time_steps: Iterable[int] = (10,25,50,100)

@dataclass
class _VaryRadius(_DefaultConfigParticles):
#   algos: Iterable[str] = ('RBF', 'KES', 'KESFast', 'KesHigher','KesHigherFast', 'SES', 'SESFast')
  algos: Iterable[str] = ('KESFast', 'SES')
  dataset__radius: Iterable[float] = (0.4,0.5,0.6,0.7)


FDARoughVol = _VaryItems(
  FDA__name = 'id',
  # dataset__nb_items = [100],
  # dataset__nb_time_steps = [5,10,25,50],
  FDA__alphas =[0.5,1,10],
  algos = ['FDA'],
  ll=None,
  at=False,
  cv = 3,
  num_trials = 5   
 )
 
SESRoughVol = _VaryItems(
  SES__depths1 =[2,3,4],
  SES__depths2 = 2,
  algos = ['SES'],
  ll=[0],
  at=True,
  cv = 3,
  num_trials = 5   
 )

KESParticles = _VaryRadius(
  KES__alphas = [10],    
  KES__rbf = True,
  KES__dyadic_order = 0,
  algos = ['KES'],
  ll=[0,1,2],
  at=True,
  cv = 3,
  num_trials = 5   
 )

KESRoughVol = _VaryItems(
  KES__alphas = [10,20],    
  KES__rbf = True,
  KES__dyadic_order = 0,
  algos = ['KES'],
  ll=[0],
  at=True,
  cv = 3,
  num_trials = 5   
 )

KESFastRoughVol = _VaryItems(
  algos = ['KESFast'],
  # dataset__nb_items = [100],
  # dataset__nb_time_steps = [5,10,25,50],
  KESFast__alphas =[1,5,10,15],
  KESFast__depths = [5],
  KESFast__ncompos = [150],
  KESFast__rbf =True,
  ll=[0],
  at=True,
  cv = 3,
  num_trials = 5
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



