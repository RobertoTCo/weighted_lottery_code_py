from time import sleep
import numpy as np
import pandas as pd

class Lottery():

    __acceptable_keys_list = ['participants_df', 'targets_df', 'weights', 'method', 'verbose', 'strict', 'scaling', 'compromise','difficulty','deterministic','sleep_time','waiting']

    def __init__(self, **kwargs): #args receives unlimited no. of arguments as an array

        # Pre-initialize attributes
        self.participants_df = None
        self.targets_df = None
        self.weights = None
        self.method = None
        self.scaling = None # Not scaling by default
        self.verbose = None # show nothing by default
        self.result_df = None
        self.strict = None # Not strict by default
        self.compromise = None # No compromise by default
        self.difficulty = None
        self.sleep_time = None # No waiting time per iteration by default
        self.deterministic = None # Not deterministic by default
        self.waiting = False # Not asking for an input to do next interation

        # Expected input attributes
        [self.__setattr__(key, kwargs.get(key)) for key in self.__acceptable_keys_list]
        
        # Check participants_df, targets_df, method and scaling are not None
        if self.participants_df is None:
            raise ValueError('The dataframe for participants must be provided')
        else:   # Avoid inplace=True altering the original dataframe
            self.participants_df = self.participants_df.copy()
        if self.targets_df is None:
            raise ValueError('The dataframe for targets must be provided')
        else:   # Avoid inplace=True altering the original dataframe
            self.targets_df = self.targets_df.copy()
        if self.method is None:
            raise ValueError("Set method argument, it must be 'harmonic' or 'linear'")
        if self.scaling is None:
            print("WARNING: Scaling is not set. The features will be aggregated without scaling. You can use 'minmax' or 'max'")
        #print(self.verbose)
        #print(self.strict)
        # Check types
        self.check_types()
        

    def __repr__(self):
        return '<Lottery demo object __repr__>'
    
    def __str__(self):
         return '<Lottery demo object __str__>'
    
    def __call__(self):
        return '<Lottery demo object __call__>'

    # -- Debugging function to check attributes that are not initialised --
    def print_none_attributes(self):
        none_attributes = []
        for attr_name, attr_value in self.__dict__.items():
            if attr_value is None:
                none_attributes.append(attr_name)
        if none_attributes:
            print("Attributes with value None:")
            for attr_name in none_attributes:
                print(f"{attr_name}: None")
        else:
            print("No attributes with value None.")
            
    # Function to check if attributes are expected types
    def check_types(self):
        if not isinstance(self.participants_df, pd.DataFrame):
            raise TypeError('The dataframe for participants must be a pandas DataFrame')
        if not isinstance(self.targets_df, pd.DataFrame):
            raise TypeError('The dataframe for targets must be a pandas DataFrame')
        if self.weights is not None:
            if not isinstance(self.weights, (list,np.ndarray)):
                raise TypeError('The weights must be a list')
            if not all(isinstance(x, (int, float)) for x in self.weights):
                raise TypeError('Each weight must be integers or floats elements')
        if not isinstance(self.method, str):
            raise TypeError('The method must be a string')
        if self.verbose is None:
            self.verbose = False
        else: 
            if not isinstance(self.verbose, bool):
                raise TypeError('The verbose must be a boolean: True or False')
        if self.strict is None:
            self.strict = False 
        else: 
            if not isinstance(self.strict, bool):
                raise TypeError('The strict must be a boolean: True or False')
        if self.scaling is not None:
            if not isinstance(self.scaling, str):
                raise TypeError("The scaling must be a string")
        if self.deterministic is None:
            print('Me cuenta NOne?')
            self.deterministic = False
        else:
            if not isinstance(self.deterministic, bool):
                raise TypeError("The deterministic input must be a boolean True or False")
        if self.waiting is None:
            self.waiting = False
        if self.compromise is not None:
            if not isinstance(self.compromise, dict):
                raise TypeError("The compromise attribute must be a dictionary with the following keys: 'compromise_vars', 'major_targets', 'minor_targets'")
            else:
                for var in ['compromise_vars', 'major_targets', 'minor_targets']:
                    if not (var in self.compromise.keys()):
                        raise ValueError(f" key {var} not in dictionary {self.compromise}. It is required")

                    if not isinstance(self.compromise[var], (list,str)):
                        raise TypeError("The dictionary keys must only have assigned a string or a list of strings as values")
                    if isinstance(self.compromise[var], list):
                        if not all(isinstance(x, str) for x in self.compromise[var]):
                            raise TypeError(f"Each element of the list {var} in '.compromise' dictionary must be a string")
                        if var == 'compromise_vars':
                            if not all(x in self.participants_df.columns for x in self.compromise[var]):
                                raise ValueError("Some elements of the list 'compromise_vars' in '.compromise' dictionary are not present in the dataframe for participants")
                        else:
                            if not all(x in self.targets_df.columns for x in self.compromise[var]):
                                raise ValueError(f"Some elements of the list {var} in the difficulty compromise are not present in the dataframe for targets")
              
        if self.difficulty is not None:
            if not isinstance(self.difficulty, dict):
                raise TypeError("The difficulty attribute must be a dictionary with the following keys: 'difficulty_vars', 'major_targets', 'minor_targets'")
            else:
                for var in ['difficulty_vars', 'major_targets', 'minor_targets']:
                    if not (var in self.difficulty.keys()):
                        raise ValueError(f" key {var} not in dictionary {self.difficulty}. It is required")

                    if not isinstance(self.difficulty[var], (list,str)):
                        raise TypeError("The dictionary keys must only have assigned a string or a list of strings as values")
                    if isinstance(self.difficulty[var], list):
                        if not all(isinstance(x, str) for x in self.difficulty[var]):
                            raise TypeError(f"Each element of the list {var} in '.difficulty' dictionary must be a string")
                        if var == 'difficulty_vars':
                            if not all(x in self.participants_df.columns for x in self.difficulty[var]):
                                raise ValueError("Some elements of the list 'compromise_vars' in '.difficulty' dictionary are not present in the dataframe for participants")
                        else:
                            if not all(x in self.targets_df.columns for x in self.difficulty[var]):
                                raise ValueError(f"Some elements of the list {var} in the difficulty dictionary are not present in the dataframe for targets")
                
    def check_compatibilties(self):
        # ! Check that the number of rows in participants_df (participants) is equal to the number of columns in the target_df.
        if self.participants_df.shape[0] != self.targets_df.shape[1]:
            if self.strict:
                raise ValueError('Strict mode selected. The number of participants must be equal to the number of targets!')
            elif self.participants_df.shape[0] < self.targets_df.shape[1]:
                print('WARNING: The number of participants is less than the number of targets.', 
                      'The lottery will be performed until all the participants are assigned to a target.')
            else:
                print('WARNING: The number of participants is greater than the number of targets.', 
                      'The lottery will be performed until all the targets are assigned to a participant.')          
        # ! Check that the number of participants variables is equal to the number of weights.
        if self.weights is not None:
            if self.targets_df.shape[1] != len(self.weights):
                raise ValueError('The number of variables for participants must be equal to the number of weights')   
    
    def preprocess_df(self):
        # ! check null & nan values for participants_df & targets_df dataframes
        if pd.isnull(self.participants_df).sum().sum() != 0:
            raise ValueError('There are missing values (NaN) in the dataframe for participants: \n\n {}'.format(self.participants_df) )
        if pd.isnull(self.targets_df).sum().sum() != 0:
            raise ValueError('There are missing values (NaN) in the dataframe for targets: \n\n {}'.format(self.targets_df) )

        # ! check column ID exists for both df or it is set as index
        if 'ID' not in self.participants_df.columns and 'ID' not in self.participants_df.index.names:
            raise ValueError('The dataframe for participants must have a column or index named ID')
        if 'ID' not in self.targets_df.columns and 'ID' not in self.targets_df.index.names:
            raise ValueError('The dataframe for targets must have a column or index named ID')
        # In case it is not set as index, set it as index
        if self.participants_df.index.name != 'ID':
            self.participants_df.set_index('ID', inplace=True)
        if self.targets_df.index.name != 'ID':
            self.targets_df.set_index('ID', inplace=True)

        # ! Check that the ids of the participants are the same in both dataframes
        if not self.participants_df.index.equals(self.targets_df.index):
            raise ValueError('The ids of the participants are not the same in both dataframes')
        # ! Check that the ids of the participants are unique in both dataframes
        if self.participants_df.index.duplicated().sum() != 0:
            raise ValueError('The ids of the participants are not unique in the dataframe for participants. Duplicated ids: ', self.participants_df.index[self.participants_df.index.duplicated()])
        if self.targets_df.index.duplicated().sum() != 0:
            raise ValueError('The ids of the participants are not unique in the dataframe for targets')
        
        # participants_df
            # ! Check that the rest of the columns are of type int or float
        my_type = ['int64', 'float64']
        dtypes = self.participants_df.dtypes.to_dict()
        for col_name, typ in dtypes.items():
            if not (typ in my_type):
                raise ValueError(f" `participants_df['{col_name}'].dtype == {typ}` not {my_type}")
        
        # targets_df
            # ! Check that the rest of the columns are of type int or float
        my_type = ['int64', 'float64']
        dtypes = self.targets_df.dtypes.to_dict()
        for col_name, typ in dtypes.items():
            if not (typ in my_type):
                raise ValueError(f" `targets_df['{col_name}'].dtype == {typ}` not {my_type}")
            # ! Check that there are unique values per row in target_df.
        if self.targets_df.duplicated().sum() != 0:
            raise ValueError('The preferences for targets must be unique per row')
            # ! Check that the values of the targets are between 1 and the number of targets.
        if not self.targets_df.isin(range(1, self.targets_df.shape[1]+1)).all().all():
            raise ValueError('The values of the targets must be between 1 and the number of targets')
        # ! Check that max(target value) = number of targets.
        if self.targets_df.max().max() != self.targets_df.shape[1]:
            raise ValueError('The max value of each target must be equal to the number of targets')
        # !Check that there are not repeated numeric values in target table for each row
        if any(self.targets_df.nunique(axis=1) < self.targets_df.shape[1]):    # if number of unique values per row is not equal to cols
            # Check which indexes of the pd.DataFrame.nunique() series are less than the number of cols
             raise ValueError(f"The rows {self.targets_df.index[self.targets_df.nunique(axis=1) < self.targets_df.shape[1]].tolist()} have repeated values")

        # check compatibilities
        self.check_compatibilties()

        # set the compromise and difficulty dicts with empty values
        # The dicts need to be declared in the case only 1 of tthe 2 dicts are specified
        if self.compromise is None:
            self.compromise = {'compromise_vars': None,'major_targets': [], 'minor_targets': []}
        if self.difficulty is None:
            self.difficulty = {'difficulty_vars': None,'major_targets': [], 'minor_targets': []}
        
    @staticmethod
    def normalize_scores(dataframe: pd.DataFrame, axis: int):
        return dataframe.div(dataframe.sum(axis=axis), axis=abs(axis-1))
    @staticmethod
    def normalize_scores_serie(dataframe: pd.DataFrame):
        return dataframe.div(dataframe.sum())
    @staticmethod
    def max_scaling(series: pd.Series):
        return series / series.max()
    @staticmethod # biased to avoid a sample is nullified in the draw because is min in all variables
    def biased_min_max_scaling(series: pd.Series):
        if series.min() == series.max():
            return series / series.max()
        else:
            return (series - series.min() + 1) / (series.max() - series.min() + 1)
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
        
    @staticmethod
    def method_harmonic_scores(dataframe: pd.DataFrame):
        # Create harmonic scores (1/value) for each preference of each participant
        scores_dataframe = dataframe.to_numpy()
        scores_dataframe = 1 / scores_dataframe
        # Transform to dataframe
        return pd.DataFrame(scores_dataframe, columns=dataframe.columns, index = dataframe.index)
    @staticmethod
    def method_linear_scores(dataframe: pd.DataFrame, axis: int):
        # Create linear scores (value) for each preference of each participant
        scores_dataframe = dataframe.to_numpy()
        scores_dataframe = dataframe.shape[axis] - scores_dataframe + 1
        # Transform to dataframe
        return pd.DataFrame(scores_dataframe, columns=dataframe.columns, index = dataframe.index)
    
    def zscores_series(self,series: pd.Series):
        return (series - series.mean()) / series.std()
    
    def check_missings(self):
        f = np.zeros(shape=(self.targets_df.shape[0],1))
        e = np.arange(1, self.targets_df.shape[1] + 1)
        for row in range(self.targets_df.shape[0]):
            v = self.targets_df.iloc[row].values
            d = np.setdiff1d(e, v)
            if d.size != 0:
                f[row,0] = d.max()
            else:
                f[row,0] = self.targets_df.shape[1]
        return f
    
    def compute_scaling(self, series: pd.Series):
        if self.scaling.lower() == "max":
            return self.max_scaling(series)
        elif self.scaling.lower() == "minmax":
            return self.biased_min_max_scaling(series)
        else:
            raise ValueError(f"Scaling method {self.scaling} not implemented")
    
    def compute_scores(self, dataframe: pd.DataFrame, axis: int):
        if self.method.lower() == "harmonic":
            return self.method_harmonic_scores(dataframe)
        elif self.method.lower() == "linear":
            return self.method_linear_scores(dataframe, axis=1)
        else:
            raise ValueError(f"Method {self.method} not implemented")
    
    def process_compromise(self):
        # If compromise method applied, compute the linear combination of the weights
        # This method must be called after the participants vars are scaled and multiplied by weights
        self.temp_compromise_vars = self.participants_df[self.compromise['compromise_vars']].copy()
        if isinstance(self.compromise['compromise_vars'],list) and len(self.compromise['compromise_vars']) > 1:
            self.temp_compromise_vars = self.temp_compromise_vars.sum(axis=0)
        # compute zscores: values below the mean are negative, above the mean are positive
            # check sd != 0
        #print(self.temp_compromise_vars)
        if np.std(self.temp_compromise_vars) == 0:
            self.temp_compromise_vars[:] = 0 # if all unique, preference not apply, sigmoid of 0 = 0.5
        else:
            self.temp_compromise_vars = self.zscores_series(self.temp_compromise_vars)
            self.temp_compromise_vars = self.temp_compromise_vars* 4 # make greater the std diff
        # compute sigmoide: negative values are less significance, positive values are more significance
        self.temp_compromise_vars = self.sigmoid(self.temp_compromise_vars)
        self.temp_compromise_vars = self.temp_compromise_vars * 2 # make mean -> 1, max -> 2

    def process_difficulty(self):
        # If difficulty method applied, compute the linear combination of the weights
        # This method must be called after the participants vars are scaled and multiplied by weights
        self.temp_difficulty_vars = self.participants_df[self.difficulty['difficulty_vars']].copy()
        if isinstance(self.difficulty['difficulty_vars'],list) and len(self.difficulty['difficulty_vars']) > 1:
            self.temp_difficulty_vars = self.temp_difficulty_vars.sum(axis=0)
        # compute zscores: values below the mean are negative, above the mean are positive
            # check sd != 0
        #print(self.temp_difficulty_vars)
        if np.std(self.temp_difficulty_vars) == 0:
            self.temp_difficulty_vars[:] = 0 # if all unique, preference not apply, sigmoid of 0 = 0.5
        else:
            self.temp_difficulty_vars = self.zscores_series(self.temp_difficulty_vars)
            self.temp_difficulty_vars = self.temp_difficulty_vars* 4 # make greater the std diff
        # compute sigmoide: negative values are less significance, positive values are more significance
        self.temp_difficulty_vars = self.sigmoid(self.temp_difficulty_vars)
        self.temp_difficulty_vars = self.temp_difficulty_vars * 2 # make mean -> 1, max -> 2

        
    def update_targets_scores(self):
        # Sort targets by score, select the highest score
        self.scores_targets_df = self.compute_scores(self.targets_df, axis=1)
        # If compromise selected, apply weights based on categories of targets
        print(f'DBG!!! Before compromise/difficulty:\n {self.scores_targets_df}')
        
        if self.compromise is not None:
            # Start calculating compromise!
            self.process_compromise()
            # Get target columns not assigned as major or minor
            # self.compromise['other_targets'] = [x for x in self.scores_targets_df.columns if x not in self.compromise['major_targets'] + self.compromise['minor_targets']]
            # Apply compromise weights for major and minor targets
            if not self.compromise['major_targets']:
                pass
            else:
                print('\n Major compromise targets not chosen yet: \n',self.scores_targets_df[self.compromise['major_targets']] )
                self.scores_targets_df[self.compromise['major_targets']] =  np.apply_along_axis(lambda x: x * self.temp_compromise_vars.to_numpy(), 0, self.scores_targets_df[self.compromise['major_targets']].to_numpy())
            if not self.compromise['minor_targets']:
                pass
            else:
                print('\n Minor compromise targets not chosen yet: \n',self.scores_targets_df[self.compromise['minor_targets']] )
                #self.scores_targets_df[self.compromise['minor_targets']] =  self.scores_targets_df[self.compromise['minor_targets']].to_numpy() * (2 - self.temp_compromise_vars.to_numpy() )
                self.scores_targets_df[self.compromise['minor_targets']] = np.apply_along_axis(lambda x: x * (2 - self.temp_compromise_vars.to_numpy()), 0, self.scores_targets_df[self.compromise['minor_targets']].to_numpy())
            print('DBG!!!!!!!!!!!!!!!!!!!!!!! temp_compromise_vars \n', self.temp_compromise_vars.to_numpy() )
            print(f'\n After compromise \n{self.scores_targets_df}')
        
        if self.difficulty is not None:
            # Start calculating difficulty!
            self.process_difficulty()
            if not self.difficulty['major_targets']:
                pass
            else:
                print('\n Major difficulty targets not chosen yet: \n',self.scores_targets_df[self.difficulty['major_targets']] )
                self.scores_targets_df[self.difficulty['major_targets']] =  np.apply_along_axis(lambda x: x * self.temp_difficulty_vars.to_numpy(), 0, self.scores_targets_df[self.difficulty['major_targets']].to_numpy())
            if not self.difficulty['minor_targets']:
                pass
            else:
                print('\n Minor difficulty targets not chosen yet: \n',self.scores_targets_df[self.difficulty['minor_targets']] )
                #self.scores_targets_df[self.difficulty['minor_targets']] =  self.scores_targets_df[self.difficulty['minor_targets']].to_numpy() * (2 - self.temp_difficulty_vars.to_numpy() )
                self.scores_targets_df[self.difficulty['minor_targets']] = np.apply_along_axis(lambda x: x * (2 - self.temp_difficulty_vars.to_numpy()), 0, self.scores_targets_df[self.difficulty['minor_targets']].to_numpy())
            print('DBG!!!!!!!!!!!!!!!!!!!!!!! temp_difficulty_vars \n', self.temp_difficulty_vars.to_numpy() )
            print(f'\n After difficulty \n{self.scores_targets_df}')
        else:
            print('Not compromise or difficulty variables applied')
            print('Punctuation for targets:\n {self.scores_targets_df}')
        #   Compute total scores for each target
        self.total_scores_targets_df = self.scores_targets_df.sum(axis=0).sort_values(ascending=False)
        #   Pick target
        if self.deterministic:
            print('\nDeterministic method')
            self.selected_target = self.total_scores_targets_df.index[0]
        else:
            print('\nProbability method')
            #   Normalise serie to range 1
            self.total_scores_targets_df = self.normalize_scores_serie(self.total_scores_targets_df)
            #   Pick target. Sample participant with probability proportional to score
            self.selected_target = np.random.choice(self.total_scores_targets_df.index, size=1, p=self.total_scores_targets_df.values)
        #   Add value to list of selected targets
        self.selected_targets.append(self.selected_target)
        if self.verbose:
            print('\nSelection of target....')
            print(f"\nTotal score for targets:\n {self.total_scores_targets_df}\n")
            print(f"\nSelected target: {self.selected_target}\n")
        
    def update_participants_scores(self):
        # Multiply selected_target preference based on participants metrics, normalise to range 1
        # and pick one participant based on probability proportional to score. 

        # Reindex participants_df to match scores_targets_df
        self.participants_df.reindex(self.scores_targets_df.index)

        # #   Column-wise scaling to 1
        # if self.scaling is not None:
        #     self.scores_participants_df = self.participants_df.apply(self.compute_scaling, axis=0)
        
        print('\nDBG!!!!', self.scores_participants_df)

        # If some columns were used to weight the preferences of targets, don't use them to compute value of participants
        # These columns  are already applyed in update_targets_scores()
        if self.compromise is not None:
            # get columns not in self.compromise['compromise_vars']
            self.scores_participants_df = self.scores_participants_df.drop(self.compromise['compromise_vars'], axis=1)
        #   aggregate element-wise sum for columns
        self.total_scores_participants_df = self.scores_participants_df.sum(axis=1)
        #   Add preferences of selected target
        #       If compromise method is applied, the preferences for the selected target are weighted based on 
        #       the compromise weights, computed in update_targets_scores()
        self.total_scores_participants_df = self.total_scores_participants_df.to_frame().join(self.scores_targets_df[self.selected_target] )
        print('\nDBG!!!! total + pais\n', self.total_scores_participants_df)
        self.total_scores_participants_df = self.total_scores_participants_df.prod(axis=1)
        #   Normalise serie to range 1
        self.total_scores_participants_df = self.normalize_scores_serie(self.total_scores_participants_df)
        #   Pick target. Sample participant with probability proportional to score
        self.selected_participant = np.random.choice(self.total_scores_participants_df.index, size=1, p=self.total_scores_participants_df.values)
        #   Add value to list of selected participants
        self.selected_participants.append(self.selected_participant)

        if self.verbose:
            print('Selection of participant....\n')
            if self.scaling is not None:
                print(f"\nScores of participants after scaling and applying weights:\n {self.scores_participants_df}")
            else:
                print(f"\nScores of participants after applying weights:\n {self.scores_participants_df}\n")
            print(f"\nProbabilities of participants in the iteration:\n {self.total_scores_participants_df}\n")
            print(f"\nSelected participant: {self.selected_participant} \n")

    def drop_selections(self):
        # Drop selections from participants_df and targets_df
        self.participants_df.drop(self.selected_participant, axis='index', inplace=True)
        self.targets_df.drop(self.selected_participant, axis='index', inplace=True)
        self.targets_df.drop(self.selected_target, axis='columns', inplace=True)
        # eliminate selected target from compromise['major_targets'] or compromise['minor_targets']
        if self.compromise is not None:
            if self.selected_target in self.compromise['major_targets']:
                self.compromise['major_targets'].remove(self.selected_target)
            elif self.selected_target in self.compromise['minor_targets']:
                self.compromise['minor_targets'].remove(self.selected_target)

    def shift_preferences(self):
        # Once a column and a row has been dropped, the values of the remaining columns and rows are not in the range 1 to the number of targets.
        # Find missing values for every row
        missing_values = self.check_missings()
        print(f'\nDBG!!!!!!!!!!!!!! Missing values: {missing_values}')
        # Reduce values greater than the missing value by 1 per row
        # This only works if the self.processing ensures every row is unique values range(1, n_cols + 1)
        self.targets_df[self.targets_df > missing_values] -= 1
        print(f'\nNew targets preferences:\n {self.targets_df}\n')
       
    def iteration(self):
        if self.verbose:
            print(f'\n  Iteration -------------> {self.iteration_number}\n')

        # 0. Scaling of participants_df
        #   Column-wise scaling to 1
        if self.scaling is not None:
            self.scores_participants_df = self.participants_df.apply(self.compute_scaling, axis=0)
        else:
            self.scores_participants_df = self.participants_df
        #   Multiply scores by the provided weights
        self.scores_participants_df = self.scores_participants_df * self.weights

        # 1. Pick: update scores for targets, pick target with highest score
        self.update_targets_scores()

        # 2. - Pick: Update scores for particpants, compute probabilities and draw a participant
            # Update scores_participant_df
        self.update_participants_scores()

        # 5. - Drop selections from participants_df and targets_df
        self.drop_selections()
        if self.targets_df.shape[1] == 0 or self.targets_df.shape[0] == 0:
            self.stop = True
        else:
        # targets preferences are shifted when 1 top target is dropped and scores recalculated
            self.shift_preferences()
            # Call sleep time to wait for the next iteration
            sleep(self.sleep_time)

        # Ask for input to continue
        if self.waiting:
            user_input = input("Press Enter to continue to the next iteration...")

    def run(self):  # Main executable function
        if self.verbose:
            print('Starting lottery algorithm!')
            print('Iteration verbose selected. Printing intermediate results. To avoid it, you can set verbose=False')
            print(f'Sleep time per iteration set to {self.sleep_time} seconds\n')

            print(f'Selected variables and targets to calculate compromise: {self.compromise}\n')

        # Preprocess dataframes
        self.preprocess_df()

        # Set non-called arguments
        if self.weights is None:
            self.weights = np.ones(self.participants_df.shape[1])

        if self.sleep_time is None:
            self.sleep_time = 0
        else:
            if not self.verbose:
                self.sleep_time = 0

        # # 0. Create dataframe with participants and empty columns for targets
        result_df = pd.DataFrame(columns=['ID','Target', 'Order'])

        # Initialize final lists for iterations
        self.selected_targets = []
        self.selected_participants = []

        print('This is your loaded tables:\n')
        print('\t Targets table:\n')
        print(self.targets_df)
        print('\t Participants table:\n')
        print(self.participants_df)
        # # Begin iterations!
        self.iteration_number = 0
        self.stop = False
        while not self.stop:
            self.iteration_number += 1
            
            print('Targets shape:',self.targets_df.shape)

            self.iteration()
        
        result_df['ID'] = self.selected_participants
        result_df['Target'] = self.selected_targets
        result_df['Order'] = np.arange(1, len(self.selected_targets) + 1)
        return result_df
