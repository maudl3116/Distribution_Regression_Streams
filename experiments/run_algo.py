import sys
import os 
sys.path.append('Distribution_Regression_Streams/src')
import KES, SES, DR_RBF  
import utils_particles, utils_roughvol
import utils
from absl import app
import configs_getter
import csv
import time 
import itertools
#TODO: iterate better config parameters
#TODO: read and plot results

_DATASETS = {
    "Particles": utils_particles.DatasetParticles, 
    "RoughVol": utils_roughvol.DatasetRoughVol,  
}

_ALGOS = {
    "KES": KES.model,            
    "KESFast": KES.model_sketch,
    "KESHigher": KES.model_higher_rank,
    "KESHigherFast": KES.model_higher_rank_sketch,
    "SES": SES.model,
    "SESFast": SES.model_sketch,
    "RBF": DR_RBF.model
}

_CSV_HEADERS = ['dataset', 'algo', 'nb_bags', 'nb_items', 'nb_time_steps', 'radius', 'mse_mean', 'mse_stdv', 'mape_mean', 'mape_stdv']

def init_seed():
  random.seed(0)
  np.random.seed(0)


def _run_algos():
  fpath = os.path.join(os.path.dirname(__file__), "../../output/metrics_draft", 
                       f'{int(time.time()*1000)}.csv')
  tmp_dirpath = f'{fpath}.tmp_results'
  os.makedirs(tmp_dirpath, exist_ok=True)
#   atexit.register(shutil.rmtree, tmp_dirpath)
  tmp_files_idx = 0

  for config_name, config in configs_getter.get_configs():          
    combinations = list(itertools.product(  config.algos, 
                                            config.dataset__name, 
                                            config.dataset__nb_bags, 
                                            config.dataset__nb_items, 
                                            config.dataset__nb_time_steps, 
                                            config.dataset__ymin, 
                                            config.dataset__ymax, 
                                            config.dataset__radius))    
    for params in combinations:                                     
        tmp_file_path = os.path.join(tmp_dirpath, str(tmp_files_idx))
        tmp_files_idx += 1
        generator = _DATASETS[params[1]](*params[2:])
        [X,y] = generator.generate()  
 
        _run_algo(tmp_file_path, [X,y], *params, config)  #TODO: need to send (1) dataset spec (2) all params for the algorithm

  print(f'Writing results to {fpath}...')                           
  with open(fpath, "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=_CSV_HEADERS)
    writer.writeheader()
    for idx in range(tmp_files_idx):
      tmp_file_path = os.path.join(tmp_dirpath, str(idx))
      try:
        with open(tmp_file_path,  "r") as read_f:
          csvfile.write(read_f.read())
      except FileNotFoundError:
        pass

  return fpath


def _run_algo(
        metrics_fpath, data, algo, dataset, nb_bags, nb_items, nb_time_steps, y_min, y_max, radius, config):
  """
  This functions runs one algo for distribution regression. It is called by _run_algos()
  which is called in main(). Below the inputs are listed which have to be
  specified in the config that is passed to main().
  Args:
   metrics_fpath: file path, automatically generated & passed by
            _run_algos()
   algo (str): the algo to train. See dict _ALGOS above.
   dividend (float): the dividend of the stock model.
  """
  print(dataset, algo, '... ', end="")
  X, y = data[0], data[1]
  # send the correct set of hyperparameters 
  if algo in ['KES']:  
      try:
        mean, stdv, results  = _ALGOS[algo](X,y, alphas=config.KES__alphas, ll=config.ll, at=config.at,  NUM_TRIALS=config.num_trials,  cv=config.cv)
      except BaseException as err:
        if fail_on_error:
            raise
        print(err)
        return
  elif algo in ["KESFast"]: 
      try:
        mean, stdv, results  = _ALGOS[algo](X,y, alphas=config.KESFast__alphas, depths=config.KESFast__depths, ncompos=config.KESFast__ncompos, 
                                            rbf=config.KESFast__rbf, ll=config.ll, at=config.at,  NUM_TRIALS=config.num_trials,  cv=config.cv)
      except BaseException as err:
        if fail_on_error:
            raise
        print(err)
        return
  elif algo in ["KESHigher"]:
      try:
        pass
        # mean, stdv, results  = _ALGOS[algo](X,y, )  #TODO: complete
      except BaseException as err:
        if fail_on_error:
            raise
        print(err)
        return
  elif algo in ["KESHigherFast"]:
      try:
        mean, stdv, results  = _ALGOS[algo](X,y,  depths1=config.KESHigherFast__depths1, ncompos1=config.KESHigherFast__ncompos1, 
                                            rbf1=config.KESHigherFast__rbf1, alphas1=config.KESHigherFast__alphas1, 
                                            lambdas_=config.KESHigherFast__lambdas_, depths2=config.KESHigherFast__depths2, ncompos2=config.KESHigherFast__ncompos2, rbf2=config.KESHigherFast__rbf2,
                                            alphas2=config.KESHigherFast__alphas2, ll=config.ll, at=config.at,  NUM_TRIALS=config.num_trials, cv=config.cv)
      except BaseException as err:
        if fail_on_error:
            raise
        print(err)
        return
  elif algo in ["SES"]:
      try:
        mean, stdv, results  = _ALGOS[algo](X,y, depths1=config.SES__depths1, depth2=config.SES__depths2, ll=config.ll, at=config.at, NUM_TRIALS=config.num_trials, cv=config.cv)
      except BaseException as err:
        if fail_on_error:
            raise
        print(err)
        return
  elif algo in ["SESFast"]: 
      try:
        mean, stdv, results  = _ALGOS[algo](X,y, depths1=config.SESFast__depths1, depths2=config.SESFast__depths2, ncompos1 = config.SESFast__ncompos1, 
                                            ncompos2 = config.SESFast__ncompos2, rbf=config.SESFast__rbf, alpha=config.SESFast__alpha, ll=config.ll, at=config.at, 
                                            NUM_TRIALS=config.num_trials, cv=config.cv)
      except BaseException as err:
        if fail_on_error:
            raise
        print(err)
        return

  mse_mean, mse_stdv = mean, stdv
  mape_mean, mape_stdv = utils.mape(results)

  metrics_ = {}
  metrics_['dataset'] = dataset
  metrics_['algo'] = algo
  metrics_['nb_bags'] = nb_bags
  metrics_['nb_items'] = nb_items
  metrics_['nb_time_steps'] = nb_time_steps
  metrics_['radius'] = radius
  metrics_['mse_mean'] = mse_mean
  metrics_['mse_stdv'] = mse_stdv
  metrics_['mape_mean'] = mape_mean
  metrics_['mape_stdv'] = mape_stdv

  with open(metrics_fpath, "w") as metrics_f:                               
    writer = csv.DictWriter(metrics_f, fieldnames=_CSV_HEADERS)
    writer.writerow(metrics_)

def main(argv):
  del argv

  try:
      filepath = _run_algos()

  except Exception as e:
    print('ERROR\n{}'.format(e))




if __name__ == "__main__":
  app.run(main)
