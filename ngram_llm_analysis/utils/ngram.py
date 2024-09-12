# ngram_trie_test.py

class TrieNode:
    __slots__ = ('children', 'count')
    
    def __init__(self):
        self.children = {}
        self.count = 0  # Count of N-grams ending at this node

class NGramTrie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, ngram):
        node = self.root
        for token in ngram:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.count += 1
    
    def keep(self, condition):
        def _keep(node, depth):
            keys_to_delete = []
            for key, child in node.children.items():
                if not _keep(child, depth + 1):
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                del node.children[key]
            return condition(node, depth) or bool(node.children)
        _keep(self.root, 0)
    
    def remove(self, condition):
        def _remove(node, depth):
            keys_to_delete = []
            for key, child in list(node.children.items()):
                if condition(child, depth):
                    keys_to_delete.append(key)
                else:
                    _remove(child, depth + 1)
            for key in keys_to_delete:
                del node.children[key]
        _remove(self.root, 0)
    
    def marginalize(self, marginalize_function):
        def _marginalize(node):
            for child in node.children.values():
                _marginalize(child)
            marginalize_function(node)
        _marginalize(self.root)
    
    def traverse_and_print(self, node=None, prefix=None):
        if node is None:
            node = self.root
            prefix = []
        if node.count > 0 and prefix:
            print(f"{' '.join(prefix)}: {node.count}")
        for token, child in node.children.items():
            self.traverse_and_print(child, prefix + [token])

def tokenize_text(data):
    """Generator that yields tokenized lines from a list of strings."""
    for line in data:
        tokens = line.strip().split()
        if tokens:
            yield tokens

def build_trie_from_dataset(data, max_n):
    """Builds an N-gram Trie from the dataset."""
    trie = NGramTrie()
    for tokens in tokenize_text(data):
        for n in range(1, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = tokens[i:i + n]
                trie.insert(ngram)
    return trie

# Example usage
if __name__ == '__main__':
    # Dataset provided by the user
    data = [
        "Together, they shared the needle and sewed the button on Lily's shirt. It was not difficult for them because they were sharing and helping each other. After they finished, Lily thanked her mom for sharing the needle and fixing her shirt. They both felt happy because they had shared and worked together.",
        "Once upon a time, there was a little car named Beep. Beep loved to go fast and play in the sun. Beep was a healthy car because he always had good fuel. Good fuel made Beep happy and strong."
    ]
    
    # Parameters
    max_n = 5  # Maximum N-gram length
    
    # Build the Trie
    print("Building the N-gram Trie...")
    trie = build_trie_from_dataset(data, max_n)
    print("Trie construction completed.\n")
    
    # Traverse and print the Trie before applying any rules
    print("Trie contents before applying rules:")
    trie.traverse_and_print()
    print("\n")
    
    # Define conditions and marginalization functions
    def remove_condition(node, depth):
        """Condition to remove N-grams that occur less than twice."""
        return node.count < 2
    
    def marginalize_function(node):
        """Aggregates counts from child nodes."""
        if node.children:
            node.count = sum(child.count for child in node.children.values())
    
    # Apply the remove rule
    print("Applying the remove rule...")
    trie.remove(remove_condition)
    print("Remove rule applied.\n")
    
    # Traverse and print the Trie after remove
    print("Trie contents after remove:")
    trie.traverse_and_print()
    print("\n")
    
    # Apply the marginalize function
    print("Applying marginalization...")
    trie.marginalize(marginalize_function)
    print("Marginalization completed.\n")
    
    # Traverse and print the Trie after marginalization
    print("Trie contents after marginalization:")
    trie.traverse_and_print()
