_all_model_ = ['GCN', 'SGC', 'GAT', 'APPNP']

def model_2_filter(model_name: str = 'GCN',
                   k_order: int = 3,
                   alpha: float = 0.1,
                   ):
    model_name = model_name.upper()
    
    if model_name in ['GCN', 'SGC']:
        filter_coeff = [0.0] * k_order
        filter_coeff[-1] = 1.0
    elif model_name in ['GAT']:
        filter_coeff = [1/k_order] * k_order
    elif model_name in ['APPNP', 'GCNII']:
        # NOTE. remove zero order coefficient and normalize
        #   coefficients from A to A^K: [α, α*(1-α), α*(1-α)^2, ..., α*(1-α)^(K-2), (1-α)^(K-1)]
        filter_coeff = [float(alpha)]
        for k in range(1, k_order-1):
            filter_coeff.append((1-alpha)*filter_coeff[-1])
        filter_coeff.append((1-alpha)**(k_order-1))
    else:
        raise ValueError(f'Invalid dataname [{model_name}].')
    return filter_coeff


if __name__ == '__main__':
    
    for model in _all_model_:
        if model in ['GCN', 'SGC', 'GAT']:
            k_order = 3
            filter_coeff = model_2_filter(model, k_order)
            print(f'Filter for [{k_order}]-order [{model}]: {filter_coeff}')
            
        elif model in ['APPNP']:
            k_order = 3
            alpha = 0.0
            filter_coeff = model_2_filter(model, k_order, alpha)
            print(f'Filter for [{k_order}]-order [{model}] (alpha=[{alpha}]): {filter_coeff}')
        
        print('-'*50)
    
    print(f'Tested All {len(_all_model_)} Datasets: {_all_model_}')