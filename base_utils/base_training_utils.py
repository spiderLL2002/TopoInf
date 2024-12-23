import torch
import torch.nn.functional as F


### Training function ###
def get_optimizer(model, args):
    if args.model=='GPRGNN':
        optimizer = torch.optim.Adam([{'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.prop1.parameters(), 'weight_decay': 0.00, 'lr': args.lr}])
    elif args.model =='BERNNET':
        optimizer = torch.optim.Adam([{'params': model.lin1.parameters(),'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    logits_train, y_train = F.log_softmax(logits[data.train_mask], dim=1), data.y[data.train_mask]
    loss = criterion(logits_train, y_train)
    loss.backward()
    optimizer.step()


def compute_masked_nodes_performance(logits, labels, criterion=None, mask=None):
    if mask == None:
        logits_masked, y_masked = logits, labels
        total_masked_nodes = len(labels)
    else:
        logits_masked, y_masked = logits[mask], labels[mask]
        total_masked_nodes = mask.sum().item()
    pred = logits_masked.max(1)[1]
    acc = pred.eq(y_masked).sum().item() / total_masked_nodes   # NOTE: `.item()` makes the code `device` independent

    masked_nodes_performance = {}
    masked_nodes_performance['acc'] = acc
    
    if criterion is not None:
        loss = criterion(logits_masked, y_masked)
        masked_nodes_performance['loss'] = loss.item()

    return masked_nodes_performance


### Evaluating function ###
@torch.no_grad()
def eval(model, data, criterion=None, get_detail=False):
    model.eval()
    logits = model(data)
    logits = F.log_softmax(logits, dim=1)

    eval_result = {}
    for key, mask in data('train_mask', 'val_mask', 'test_mask'):
        eval_result[key] = compute_masked_nodes_performance(logits=logits, labels=data.y, criterion=criterion, mask=mask)
        if get_detail:
            eval_result[key]['masking'] = data[key]
    
    if get_detail:
        eval_result['all_mask'] = compute_masked_nodes_performance(logits=logits, labels=data.y, criterion=criterion, 
                                                                mask=data.train_mask|data.val_mask|data.test_mask)   # train + val + test
        eval_result['all_nodes_except_train'] = compute_masked_nodes_performance(logits=logits, labels=data.y, criterion=criterion, 
                                                               mask=~data.train_mask)   # all nodes except train
        eval_result['all_nodes'] = compute_masked_nodes_performance(logits=logits, labels=data.y, criterion=criterion, 
                                                                mask=None)   # all nodes
        eval_result['all_nodes']['logits'] = logits.cpu()
    
    return eval_result

@torch.no_grad() 
def calculate_entropy(model, data):
    model.eval()
    logits = model(data)
    logits = F.softmax(logits, dim=1)
    
    entropy = -torch.sum(logits * torch.log(logits ), dim=1) 
    entropy_dict = {}
    for node_id, ent in enumerate(entropy):
        entropy_dict[node_id] = ent.item()  
        
    '''print("Logits min: ", logits.min(), " Logits max: ", logits.max())
    print("Entropy min and max",min(entropy_dict.values()) ,max(entropy_dict.values())) 
    '''
    return entropy_dict
    
    
    
def eval_pred(pred, data, criterion=None, detailed=False):
    eval_result = {}
    for key, mask in data('train_mask', 'val_mask', 'test_mask'):
        eval_result[key] = compute_masked_nodes_performance(logits=pred, labels=data.y, criterion=criterion, mask=mask)

    if detailed:
        eval_result['all_mask'] = compute_masked_nodes_performance(logits=pred, labels=data.y, criterion=criterion, 
                                                                mask=data.train_mask|data.val_mask|data.test_mask)   # train + val + test
        eval_result['all_nodes'] = compute_masked_nodes_performance(logits=pred, labels=data.y, criterion=criterion, 
                                                                mask=None)   # all nodes
        eval_result['all_nodes_except_train'] = compute_masked_nodes_performance(logits=pred, labels=data.y, criterion=criterion, 
                                                                mask=~data.train_mask)   # all nodes except train
    
    return eval_result




def print_eval_result(eval_result, prefix=''):
    if prefix:
        prefix = prefix + ' '
    print(f"{prefix}"
        f"Train Acc:{eval_result['train_mask']['acc']*100:6.2f} | "
        f"Val Acc:{eval_result['val_mask']['acc']*100:6.2f} | "
        f"Test Acc:{eval_result['test_mask']['acc']*100:6.2f}")






