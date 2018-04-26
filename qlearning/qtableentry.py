import hashlib


class QTableEntry(object):
    def __init__(self, state):
        self.__state_str = str(state).encode('utf-8')

        hash_object = hashlib.sha256(self.__state_str)

        self.__hash = hash_object.hexdigest()
        self.__state = state
        self.__q_values = []
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
