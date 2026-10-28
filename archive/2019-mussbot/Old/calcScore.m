function [score, diceRemaining, mask_busts] = calcScore(dice)
% CALCSCORE
%
%	INPUT
%
%
%
%	OUTPUT
%
%
% John W. Miller
% 05-Mar-2017

muss_buss_rules = false;

% Pre allocate
n_rolls = size(dice,1);
if n_rolls == 1
    n_rolls = 2;
    dice = [dice; zeros(1,length(dice))];
end
score = zeros(n_rolls,1);

% Dice values
values_indiv = [100 0 0 0 50 0];
values_trips = [1000 200 300 400 500 600];

% Count occurrence of each number for all rolls
counts = zeros(n_rolls,6);
for n_side = 1:6
    counts(:,n_side) = sum(bsxfun(@eq,dice,n_side),2);
end

% Calculate the total points for each roll
if ~muss_buss_rules % Traditional Fill or Bust rules    
    % Check for straights (and remove those dice from the count)
    score( all(counts==1,2)) = 1500;
    counts(all(counts==1,2),:) = 0;
    
    % Check for triples (and remove those dice from the count)
    for n = 1:2 % Do it twice incase we have six of a kind
        score = score + sum(bsxfun(@mtimes,counts>=3,values_trips),2);
        counts = max(counts-3*(counts>=3),0);
    end
    
    % Check for ones and fives
    score = score + sum(bsxfun(@mtimes,counts,values_indiv),2);
    counts(:,[1 5]) = 0;    
        
else % MussBuss Rules
    
    % Remember to account for the six of a kind auto-win! :)
    
    
end

% Keep track of remaining dice and busts
diceRemaining = sum(counts,2);
diceRemaining(score==0) = 0; % Busts
mask_busts = and(score==0,~all(isnan(dice),2));

% If it's only a single roll
if all(dice(end,:))==0        
    [score, diceRemaining, mask_busts] = deal(score(1,:),diceRemaining(1),mask_busts(1));            
end


end % End of main