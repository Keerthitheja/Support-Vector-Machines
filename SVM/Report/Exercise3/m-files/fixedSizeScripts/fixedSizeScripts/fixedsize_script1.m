clear
close all


X = 3.*randn(100,2);
ssize = 10;
sigList = [0.001 ];
subset = zeros(ssize,2);
for sig2 = sigList
    figure
    for t = 1:100,
        %
        % new candidate subset
        %
        r = ceil(rand*ssize);
        candidate = [subset([1:r-1 r+1:end],:); X(t,:)];
  
        %
        % is this candidate better than the previous?
        %
        if kentropy(candidate, 'RBF_kernel',sig2)>...
            kentropy(subset, 'RBF_kernel',sig2),
            subset = candidate;
        end
  
        %
        % make a figure
        %
        plot(X(:,1),X(:,2),'b*','MarkerSize',15, 'linewidth',2); hold on;
        plot(subset(:,1),subset(:,2),'ro','MarkerSize',15,'linewidth',6); hold off; 
        pause(0.1)
    end
end