Node:
    Index:  identifier to node
    Attribute: Attribute on which the node is splitting
    isLeaf: Boolean indicating if the node is a leaf or not
    Children: List of children nodes
    subdata: ?

    addChild(self, child, value, symbol): Adds a child to the node with given value of attribute with respect to the symbol

    createLeaf(self, target): Creates a leaf node with given target value as output

    leaf node: leaf node is node with a target value and data same as given by parent node

DecisionTree:
    train_data: Training data
    test_data: Testing data
    validation_data: Validation data
    root: Root node of the tree
    train_pred: Predictions on training data
    test_pred: Predictions on testing data
    validation_pred: Predictions on validation data

    get_data(): Returns the training, testing, and validation data
    train(): Call build tree function and saves as root
    _build_tree(data,max_depth):
    
    choose_best_feature(data):  