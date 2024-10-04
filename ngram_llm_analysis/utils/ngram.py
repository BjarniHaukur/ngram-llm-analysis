import json
from pathlib import Path

CHECKPOINT_PATH = Path("../checkpoints/ngram/")
CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

class TrieNode:
    __slots__ = ('children', 'count')
    
    def __init__(self):
        self.children:dict[int, TrieNode] = {} 
        self.count:int = 0  # Count of N-grams ending at this node

    def to_dict(self):
        return {
            'children': {k: v.to_dict() for k, v in self.children.items()},
            'count': self.count
        }

    @classmethod
    def from_dict(cls, data):
        node = cls()
        node.children = {k: cls.from_dict(v) for k, v in data['children'].items()}
        node.count = data['count']
        return node

class NGramTrie:
    """ A trie for storing N-grams of tokenized text. """
    def __init__(self, ngram_max_length:int):
        self.root = TrieNode()
        self.ngram_max_length = ngram_max_length  # not used
    
    def save(self, filename:str):
        filepath = CHECKPOINT_PATH / (filename if filename.endswith(".json") else filename + ".json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'ngram_max_length': self.ngram_max_length,
                'root': self.root.to_dict()
            }, f, indent=2)
    
    @classmethod
    def load(cls, filename:str):
        filepath = CHECKPOINT_PATH / (filename if filename.endswith(".json") else filename + ".json")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        trie = cls(data['ngram_max_length'])
        trie.root = TrieNode.from_dict(data['root'])
        return trie
    
    @classmethod
    def fit(cls, tokens:list[int], ngram_max_length:int):
        trie = cls(ngram_max_length)
        for i in range(len(tokens) - ngram_max_length + 1): # sliding window over tokens
            ngram = tokens[i:i+ngram_max_length]
            trie.insert(ngram)
        return trie
    
    def insert(self, ngram):
        assert len(ngram) == self.ngram_max_length, "N-gram length must be equal to the maximum length"
        node = self.root
        for token in ngram:
            node.children[token] = node.children.get(token, TrieNode()) # get or create the node
            node = node.children[token] # move to the next node
            node.count += 1 # increment the count
        
    def find_all_nodes(self, tokens:list[int], rule_context:str|None=None)->list[TrieNode]:
        def recursive_search(node:TrieNode, rule_context:str, index:int)->list[TrieNode]:
            if index == len(rule_context): return [node]  # reached the end of the rule context, return this node

            if (token:= rule_context[index]) == '*':
                results = []
                for child_node in node.children.values(): # try out all wildcards
                    results.extend(recursive_search(child_node, rule_context, index + 1))
                return results
            elif token in node.children:
                return recursive_search(node.children[token], rule_context, index + 1)
            else:  # not found 
                return []

        search_context = self._preprocess_rule_context(tokens, rule_context)
        return recursive_search(self.root, search_context, 0)
    
    def _preprocess_rule_context(self, tokens:list[int], rule_context:str|None)->str:
        if rule_context is None: return tokens
        assert len(tokens) == len(rule_context), "Tokens and rule context must be of the same length"
        return ["*" if rule == "*" else token for token, rule in zip(tokens, rule_context) if rule != "-"]
    
    def search(self, tokens:list[int], rule_context:str|None=None)->int:
        """
        Search the trie using tokens and rule context. '-' shortens context, '*' is wildcard and '+' is the keep.
        
        Example:
            search([1,2,3], "*-+") searches anything (*) followed by 3
            
            search([1,2,3], "--*") searches for all ngrams of length 1
            
            search([1,2,3], None) is equivalent to search([1,2,3], "+++"), i.e. search for this exact ngram
        """
        def _search(search_context:str, node:TrieNode=None, index:int=0)->int:
            node = node or self.root  # start at root unless specified

            if index == len(search_context): return node.count  # Reached the end of the rule context

            total_count = 0
            
            if (token:= search_context[index]) == '*':  # explore all child nodes at this position
                total_count += sum(_search(search_context, child_node, index + 1) for child_node in node.children.values())
            elif token in node.children:  # continue search
                total_count += _search(search_context, node.children[token], index + 1)
            else:  # not found
                return 0

            return total_count
        
        search_context = self._preprocess_rule_context(tokens, rule_context)
        return _search(search_context)
    
    def unique_successor_count(self, context:list[int], context_rule:str|None=None)->int:
        unique_successors = set()
        for node in self.find_all_nodes(context, context_rule):
            unique_successors.update(node.children.keys())
        return len(unique_successors)
    
    def continuation_count(self, token: int) -> int:
        """Count the number of unique contexts that precede the given token."""
        unique_contexts = set()
        stack = [(self.root, [])]
        while stack:
            current_node, context = stack.pop()
            for child_token, child_node in current_node.children.items():
                new_context = context + [child_token]
                if child_token == token and context:
                    unique_contexts.add(tuple(context))
                stack.append((child_node, new_context))
        return len(unique_contexts)
    
    def total_unique_contexts(self, n:int)->int:
        """Compute the total number of unique (n-1)-gram contexts."""
        count = 0
        stack = [(self.root, [])]  # Stack holds (current_node, path)
        
        while stack:
            current_node, path = stack.pop()

            if len(path) == n - 1:  # We only count paths of length (n-1)
                count += 1
                continue  # no need to go deeper for this path

            # then the path is shorter so we continue exploring
            for child_token, child_node in current_node.children.items():
                stack.append((child_node, path + [child_token]))
        
        return count
    
    def kneser_ney_smoothed_ratios(self, tokens:list[int], rule_context:str, discount:float=0.75)->list[float]:
        """Compute Kneser-Ney smoothed probability ratios for the given tokens and rule context."""
        ratios = []
        total_continuation = self.total_unique_contexts(len(tokens))

        for i in range(len(tokens)):
            context = tokens[:i]
            context_rule = rule_context[:i]
            token = tokens[i]

            ngram_count = self.search(context + [token], context_rule + "+")
            context_count = self.search(context, context_rule)
            unique_successors = self.unique_successor_count(context, context_rule)
            continuation_count = self.continuation_count(token)

            if context_count > 0:
                discounted_prob = max(ngram_count - discount, 0) / context_count
                lambda_factor = (discount * unique_successors) / context_count
                backoff_prob = continuation_count / total_continuation
                kn_prob = discounted_prob + lambda_factor * backoff_prob
                ratios.append(kn_prob)
            else:
                backoff_prob = continuation_count / total_continuation
                ratios.append(backoff_prob)

        return ratios
    
    def predict_next_token(self, context:list[int], rule_context:str|None=None)->int:
        """Predict the next token in the sequence using Kneser-Ney smoothed probabilities."""
        rule_context = rule_context or "+" * len(context)
        ratios = self.kneser_ney_smoothed_ratios(context, rule_context)
        max_ratio = max(ratios)
        return ratios.index(max_ratio)
        
            
if __name__ == "__main__":
    trie = NGramTrie(ngram_max_length=3)
    trie.insert([1, 2, 3])
    trie.insert([1, 2, 3])
    trie.insert([1, 2, 4])
    trie.insert([2, 3, 4])
    trie.insert([3, 4, 5])

    # Test insertion and counts
    count = trie.search([1, 2, 3], "+++")
    assert count == 2, f"Test failed: Incorrect count for [1, 2, 3], expected 2 got {count}"
    count = trie.search([1, 2, 4], "+++")
    assert count == 1, f"Test failed: Incorrect count for [1, 2, 4], expected 1 got {count}"

    # Test search with rule context
    count = trie.search([1, 2, 3], "*++")
    assert count == 2, f"Test failed: Incorrect count for '* 2 3', expected 2 got {count}"
    count = trie.search([1, 2, 3], "+*+")
    assert count == 2, f"Test failed: Incorrect count for '1 * 3', expected 2 got {count}"
    count = trie.search([1, 2, 3], "++*")
    assert count == 3, f"Test failed: Incorrect count for '1 2 *', expected 3 got {count}"
    count = trie.search([1, 2, 3], "**+")
    assert count == 2, f"Test failed: Incorrect count for '* * 3', expected 2 got {count}"

    # Test different context lengths
    count = trie.search([1, 2, 3], "-++")
    assert count == 1, f"Test failed: Incorrect count for context '-++', expected 2 got {count}"
    count = trie.search([1, 2, 3], "--+")
    assert count == 1, f"Test failed: Incorrect count for context '--+', expected 2 got {count}"

    # Test edge cases
    empty_trie = NGramTrie(ngram_max_length=3)
    count = empty_trie.search([1, 2, 3], "+++")
    assert count == 0, f"Test failed: Empty trie should return 0, got {count}"
    count = trie.search([9, 9, 9], "+++")
    assert count == 0, f"Test failed: Non-existent n-gram should return 0, got {count}"

    # Test unique_successor_count
    unique_successors = trie.unique_successor_count([1, 2], "++")
    assert unique_successors == 2, f"Test failed: Expected 2 unique successors for context [1, 2], got {unique_successors}"

    # Test total_unique_contexts
    total_contexts = trie.total_unique_contexts(3)
    assert total_contexts == 3, f"Test failed: Expected 3 total unique contexts of length 2, got {total_contexts}"

    # Test corrected continuation_count
    continuation_count_3 = trie.continuation_count(3)
    assert continuation_count_3 == 2, f"Test failed: Expected 2 unique contexts preceding token 3, got {continuation_count_3}"

    # Test kneser_ney_smoothed_ratios
    tokens = [1, 2, 3]

    kn_ratios = trie.kneser_ney_smoothed_ratios(tokens, rule_context, discount=0.75)

    # Expected Kneser-Ney probability for the last token
    expected_kn_prob = 0.75  # Calculated based on earlier verification

    # Since kn_ratios is a list of probabilities for each token, we check the last one
    last_kn_prob = kn_ratios[-1]
    assert abs(last_kn_prob - expected_kn_prob) < 1e-5, f"Test failed: Expected KN probability {expected_kn_prob}, got {last_kn_prob}"

    print("All tests passed successfully.")