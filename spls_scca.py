import numpy as np


def softthresh(x,mu):
    return np.sign(x) * np.maximum(np.abs(x) - mu, 0)

def witten(X, Y=None, lx=0, ly=0, max_iter=100, verbose=0, tol=1e-6):
    if Y is None:
        A = X
    else:
        A = X.T @ Y

    u,s,v = np.linalg.svd(A, full_matrices=False)
    u = u[:,:1]
    v = v[:1,:].T

    loss = np.zeros(max_iter)
    
    for iter in range(max_iter):
        u = softthresh(A @ v, lx)
        if np.linalg.norm(u) > 0:
            u = u / np.linalg.norm(u)
        v = softthresh(A.T @ u, ly)
        if np.linalg.norm(v) > 0:
            v = v / np.linalg.norm(v)
            
        loss[iter] = u.T @ A @ v
        
        if iter > 0 and np.abs(loss[iter]-loss[iter-1]) < tol:
            if verbose > 0:
                print('Converged in {} iteration(s)'.format(iter))
            break
        if (iter == max_iter-1) and (verbose > 0):
            print('Did not converge. Losses: ', loss)

    if np.sum(v) < 0:
        v = -v
        u = -u

    return (u,v)

def witten_cv(X, Y, reg_params, reps=10, folds=10, seed=42, ncomps=1):
    n = X.shape[0]
    testcorrs = np.zeros((folds, reps, len(reg_params), ncomps))
    nonzero = np.zeros((folds, reps, len(reg_params), ncomps))

    # CV repetitions
    np.random.seed(seed)
    for rep in range(reps):
        print('.', end='')
        ind = np.random.permutation(n)
        X = X[ind,:]
        Y = Y[ind,:]
        
        # CV folds
        for cvfold in range(folds):
            indtest  = np.arange(cvfold*int(n/folds), (cvfold+1)*int(n/folds))
            indtrain = np.setdiff1d(np.arange(n), indtest)
            Xtrain = np.copy(X[indtrain,:])
            Ytrain = np.copy(Y[indtrain,:])
            Xtest  = np.copy(X[indtest,:])
            Ytest  = np.copy(Y[indtest,:])
            
            # mean centering
            X_mean = np.mean(Xtrain, axis=0)
            Xtrain -= X_mean
            Xtest  -= X_mean
            Y_mean = np.mean(Ytrain, axis=0)
            Ytrain -= Y_mean
            Ytest  -= Y_mean

            # loop over regularization parameters
            for i,r in enumerate(reg_params):    
                vx,vy = witten(Xtrain, Ytrain, lx=r)
                
                if (np.sum(vx!=0)==0) or (np.sum(vy!=0)==0):
                    nonzero[cvfold, rep, i, 0] = np.nan
                    continue
                
                testcorrs[cvfold, rep, i, 0] = np.corrcoef((Xtest @ vx).T, (Ytest @ vy).T)[0,1]
                nonzero[cvfold, rep, i, 0] = np.sum(vx!=0)

                # loop over components (deflating)
                A = Xtrain.T @ Ytrain
                for ncomp in range(1, ncomps):
                    d = vx.T @ A @ vy
                    A = A - d * vx @ vy.T
                    vx, vy = witten(A, lx = r)
                    if (np.sum(vx!=0)==0) or (np.sum(vy!=0)==0):
                        nonzero[cvfold, rep, i, ncomp] = np.nan
                        continue
                    testcorrs[cvfold, rep, i, ncomp] = np.corrcoef((Xtest @ vx).T, (Ytest @ vy).T)[0, 1]
                    nonzero[cvfold, rep, i, ncomp] = np.sum(vx!=0)

    print(' done')
    return np.squeeze(testcorrs), np.squeeze(nonzero)

def witten_bootstrap(X, Y, lx=0, ly=0, nrep = 100, seed=42):
    np.random.seed(seed)
    ww = np.zeros((X.shape[1], nrep))
    for rep in range(nrep):
        print('.', end='')
        n = np.random.choice(X.shape[0], size = X.shape[0])
        w,v = witten(X[n,:], Y[n,:], lx=lx, ly=ly)
        ww[:,rep] = w[:,0]
    print(' done')
    bootCounts = np.sum(ww!=0, axis=1)/nrep
    return bootCounts


def linadmm(a, tau, A = None, lambd=1, mu=1, x0 = None, ridge=0, tol=1e-10):
    if x0 is None:
        x0 = np.zeros_like(a)
    if A is None:
        A = np.eye(np.size(a))
    
    x = x0
    z = np.zeros(A.shape[0])
    u = np.zeros(A.shape[0])
    for i in range(100):
        xold = np.copy(x)
        x = softthresh((x - mu/lambd * A.T @ (A @ x - z + u) + mu*a)/(1+mu*ridge), mu*tau/(1+mu*ridge))
        z = A @ x + u
        if np.linalg.norm(z) > 1:
            z = z / np.linalg.norm(z)
        u = u + A @ x - z
        if np.sum((x-xold)**2) < tol:
#             print(i, 'admm iterations')
            break
    return x
        

def suo(X, Y, lx=0, ly=0, max_iter=100, verbose=0, tol=1e-6):
    A = X.T @ Y
    u,s,v = np.linalg.svd(A, full_matrices=False)
    u = u[:,0]
    v = v[0,:].T
    u = u/np.linalg.norm(X@u)
    v = v/np.linalg.norm(Y@v)
    
    maxEigX = np.max(np.linalg.svd(X, compute_uv=False))
    maxEigY = np.max(np.linalg.svd(Y, compute_uv=False))
    
    loss = np.zeros(max_iter)
    
    for iter in range(max_iter):
        # should coincide with Witten if None is used instead of X and 1 instead of maxEig
        u = linadmm(A @ v,   lx, X, 1, 1/maxEigX**2, x0 = u)  
        v = linadmm(A.T @ u, ly, Y, 1, 1/maxEigY**2, x0 = v)
        
        loss[iter] = u.T @ A @ v
        
        if iter > 0 and np.abs(loss[iter]-loss[iter-1]) < tol:
            if verbose > 0:
                print('Converged in {} iteration(s)'.format(iter))
            break
        if (iter == max_iter-1) and (verbose > 0):
            print('Did not converge. Losses: ', loss)

    if np.sum(v) < 0:
        v = -v
        u = -u

    return (u[:,np.newaxis], v[:,np.newaxis])

def suo_cv(X, Y, reg_params, reps=10, folds=10, seed=42):
    n = X.shape[0]
    testcorrs = np.zeros((folds, reps, len(reg_params)))
    nonzero = np.zeros((folds, reps, len(reg_params)))

    # CV repetitions
    np.random.seed(seed)
    for rep in range(reps):
        print('.', end='')
        ind = np.random.permutation(n)
        X = X[ind,:]
        Y = Y[ind,:]
        
        # CV folds
        for cvfold in range(folds):
            indtest  = np.arange(cvfold*int(n/folds), (cvfold+1)*int(n/folds))
            indtrain = np.setdiff1d(np.arange(n), indtest)
            Xtrain = np.copy(X[indtrain,:])
            Ytrain = np.copy(Y[indtrain,:])
            Xtest  = np.copy(X[indtest,:])
            Ytest  = np.copy(Y[indtest,:])
            
            # mean centering
            X_mean = np.mean(Xtrain, axis=0)
            Xtrain -= X_mean
            Xtest  -= X_mean
            Y_mean = np.mean(Ytrain, axis=0)
            Ytrain -= Y_mean
            Ytest  -= Y_mean

            # loop over regularization parameters
            for i,r in enumerate(reg_params):    
                vx,vy = suo(Xtrain, Ytrain, lx=r)
                
                if (np.sum(vx!=0)==0) or (np.sum(vy!=0)==0):
                    nonzero[cvfold, rep, i] = np.nan
                    continue
                    
                testcorrs[cvfold, rep, i] = np.corrcoef((Xtest @ vx).T, (Ytest @ vy).T)[0,1]
                nonzero[cvfold, rep, i] = np.sum(vx!=0)
    
    print(' done')
    return testcorrs, nonzero
