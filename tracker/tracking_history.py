import numpy as np
from scipy.optimize import linear_sum_assignment

class HistoryKeeper:
    def __init__(self, past_detections = None) -> None:
        self.history = dict()
        if past_detections is not None:
            for frame in past_detections:
                self.update(frame)
                
    def update(self, frame, assignment_algorithm):
        '''
        Here we take a given frame and update our history. 
        This means we take the detections and try to match with any of the previously detected IDs.
        The flow is the following:
        frame_detection -> IoU matching -> Use algorithm to assign ID
        
        This algorithm can be a number of options:
        -> Greedy
        -> Hungarian
        -> Markov Chain Monte Carlo
        -> Distance Based Methods
        -> Deep Learning (most costly)
        -> Etc.
        
        The algorithm will be assignable, so that multiple can be tested at once.
        '''
        # Match detections to existing tracks
        if len(self.history) > 0:
            # Get IDs and detections from the last frame in the history
            last_ids = list(self.history.keys())
            last_detections = [self.history[id][-1] for id in last_ids]
            
            # Compute IoU between detections and last frame detections
            iou_matrix = np.zeros((len(frame), len(last_detections)))
            for i, detection in enumerate(frame):
                for j, last_detection in enumerate(last_detections):
                    iou_matrix[i, j] = self.compute_iou(detection[:4], last_detection[:4])
            
            # Run the assignment algorithm to match detections to tracks
            matches, unmatched_detections, unmatched_tracks = self.run_assignment_algorithm(iou_matrix, assignment_algorithm)
            
            # Update existing tracks with matched detections
            for row, col in matches:
                id = last_ids[col]
                detection = frame[row]
                self.history[id].append(detection)
            
            # Create new tracks for unmatched detections
            for i in unmatched_detections:
                detection = frame[i]
                id = detection[5]
                self.history[id] = [detection]
        else:
            # Create tracks for all detections in the first frame
            for i, detection in enumerate(frame):
                id = detection[5]
                self.history[id] = [detection]
                
    def run_assignment_algorithm(self, iou_matrix, algorithm):
        '''
        Run the specified assignment algorithm on the IoU matrix and return the matched detections and tracks.
        '''
        if algorithm == "greedy":
            return self.greedy_match(iou_matrix)
        elif algorithm == "hungarian":
            return self.hungarian_match(iou_matrix)
        elif algorithm == "mcmc":
            return self.markov_chain_monte_carlo_match(iou_matrix)
        elif algorithm == "distance":
            return self.distance_match(iou_matrix)
        else:
            raise ValueError("Invalid assignment algorithm specified.")
    
    def greedy_match(self, iou_matrix):
        '''
        Greedy matching algorithm based on maximum IoU.
        '''
        num_detections, num_tracks = iou_matrix.shape
        matches = []
        unmatched_detections = []
        unmatched_tracks = list(range(num_tracks))
        
        # Step 1: Assign detections to tracks based on maximum IoU
        for i in range(num_detections):
            max_iou = 0
            max_idx = None
            for j in unmatched_tracks:
                iou = iou_matrix[i, j]
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            if max_idx is not None:
                matches.append((i, max_idx))
                unmatched_tracks.remove(max_idx)
            else:
                unmatched_detections.append(i)
        
        # Step 2: Handle unmatched tracks
        if len(unmatched_tracks) > 0:
            # Find the detection with the maximum IoU for each unmatched track
            for j in unmatched_tracks:
                max_iou = 0
                max_idx = None
                for i in unmatched_detections:
                    iou = iou_matrix[i, j]
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = i
                if max_idx is not None:
                    matches.append((max_idx, j))
                    unmatched_detections.remove(max_idx)
        
        return matches, unmatched_detections, unmatched_tracks

    
    def hungarian_match(self, iou_matrix):
        '''
        Hungarian matching algorithm based on linear_sum_assignment from scipy.
        '''
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matches = [(row_ind[i], col_ind[i]) for i in range(len(row_ind))]
        unmatched_detections = [i for i in range(iou_matrix.shape[0]) if i not in row_ind]
        unmatched_tracks = [j for j in range(iou_matrix.shape[1]) if j not in col_ind]
        return matches, unmatched_detections, unmatched_tracks
    
    def markov_chain_monte_carlo_match(self, iou_matrix, num_iterations=1000, initial_state=None, temperature=1.0):
        '''
        Markov Chain Monte Carlo matching algorithm based on IoU values.

        Parameters:
        iou_matrix (numpy.ndarray): NxM array containing IoU values between N detections and M tracks.
        num_iterations (int): Number of iterations for the MCMC simulation.
        initial_state (list): List of initial states for the MCMC simulation. Default is None (random initial states).
        temperature (float): Temperature parameter for the Metropolis-Hastings algorithm. Default is 1.0 (no scaling).

        Returns:
        matches (list): List of matched detections and tracks.
        unmatched_detections (list): List of unmatched detections.
        unmatched_tracks (list): List of unmatched tracks.
        '''
        num_detections, num_tracks = iou_matrix.shape
        if initial_state is None:
            # Generate random initial state
            initial_state = list(range(num_detections))

        # Define transition matrix based on IoU values
        transition_matrix = np.zeros((num_detections, num_detections))
        for i in range(num_detections):
            for j in range(num_detections):
                transition_matrix[i, j] = np.exp((iou_matrix[i, j] - 1) / temperature)

        # Normalize rows to obtain probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix /= row_sums[:, np.newaxis]

        # Initialize MCMC simulation
        current_state = initial_state
        current_score = self.compute_assignment_score(iou_matrix, current_state)
        best_state = current_state
        best_score = current_score

        # Run MCMC simulation
        for i in range(num_iterations):
            # Propose a new state by randomly swapping two detections
            new_state = current_state.copy()
            j, k = np.random.choice(num_detections, size=2, replace=False)
            new_state[j], new_state[k] = new_state[k], new_state[j]

            # Compute the score of the new state
            new_score = self.compute_assignment_score(iou_matrix, new_state)

            # Accept or reject the new state based on the Metropolis-Hastings algorithm
            acceptance_prob = min(1, np.exp((new_score - current_score) / temperature))
            if np.random.rand() < acceptance_prob:
                current_state = new_state
                current_score = new_score

            # Update the best state if necessary
            if current_score > best_score:
                best_state = current_state
                best_score = current_score

        # Generate matches based on the best state
        matches = [(i, best_state[i]) for i in range(num_detections)]
        matches = [match for match in matches if iou_matrix[match[0], match[1]] > 0]
        unmatched_detections = [i for i in range(num_detections) if i not in best_state]
        unmatched_tracks = [i for i in range(num_tracks) if i not in best_state]

        return matches, unmatched_detections, unmatched_tracks
