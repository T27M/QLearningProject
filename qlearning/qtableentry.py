import hashlib
import random


class QTableEntry(object):
    def __init__(self, state):
        self.__state_str = str(state).encode('utf-8')

        hash_object = hashlib.sha256(self.__state_str)

        self.__hash = hash_object.hexdigest()
        self.__state = state

        self.__q_values = {
            'UP': 0.0,
            'LEFT': 0.0,
            'RIGHT': 0.0,
            'DOWN': 0.0
        }

        self.__next_state = None
        pass

    def get_hash(self):
        return self.__hash

    def get_state(self):
        return self.__state

    def get_state_str(self):
        return self.__state_str

    def compare_hash(self, other):
        return self.__hash == other.get_hash()

    def compare_state(self, other):
        return self.__state_str == other.get_state_str()

    def get_q_values(self):
        return self.__q_values

    def get_q_value(self, action):
        return self.__q_values[action]

    def set_q_value(self, action, value):
        self.__q_values[action] = value

    def get_q_value_max(self):
        """ Finds the max qvalue from the qtable for all actions

        Returns:
            float -- the qvalue
        """
        max_val = max(self.__q_values.values())
        keys = (k for k, v in self.__q_values.items() if v == max_val)

        return self.get_q_value(self.__iter_sample_fast(keys, 1)[0])

    def has_next_vector(self):
        """ Used to check for a hash collision

        Returns:
            bool -- True if contains multiple vectors
        """
        return self.__next_state is not None

    def next_vector(self):
        """ Attempts to get the next vector

        Raises:
            ValueError -- if next vector is None

        Returns:
            QTableEntry -- the next qtableentry
        """
        if(self.__next_state is None):
            raise ValueError(
                "QTable entry does not contain another vector value")

        return []

    def add_next(self, q_table_entry):
        """ Adds a new entry to create a linked list of entries under a hash

        Arguments:
            q_table_entry {QTableEntry} -- the collided entry to add

        Raises:
            ValueError -- hash value do not match
            ValueError -- vector values are identical
        """

        # Ensure q_table_entries have collided
        if(not self.compare_hash(q_table_entry)):
            raise ValueError("QTableEntry hash values do not match")

        # Ensure that the states are not the same
        if(self.compare_state(q_table_entry)):
            raise ValueError("Next QTableEntry must not be identical")

        self.__next_state = q_table_entry

    # https://stackoverflow.com/questions/12581437/python-random-sample-with-a-generator-iterable-iterator?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    def __iter_sample_fast(self, iterable, samplesize):
        results = []
        iterator = iter(iterable)
        # Fill in the first samplesize elements:
        try:
            for _ in range(samplesize):
                results.append(next(iterator))
        except StopIteration:
            raise ValueError("Sample larger than population.")
        random.shuffle(results)  # Randomize their positions
        for i, v in enumerate(iterator, samplesize):
            r = random.randint(0, i)
            if r < samplesize:
                results[r] = v  # at a decreasing rate, replace random items
        return results
