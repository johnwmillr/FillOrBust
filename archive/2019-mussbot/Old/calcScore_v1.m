function [scores, n_dice_to_roll_next, mask_straights] = calcScore_v1(dice)
% CALCSCORE 
%
%	INPUT
%
%
%	OUTPUT
%
% John W. Miller
% 09-Jan-2017

% Point values
values_singles = [100 0 0 0 50 0];
values_triples = [1000 200 300 400 500 600];

% Deal w/ the case of a single roll
if size(dice,1) == 1
    dice = [dice ; nan(1,6)];    
    single_roll = true;
else
    single_roll = false;
end, n_rolls = size(dice,1);     

% Count occurrence of each number for all rolls
counts = zeros(n_rolls,6);
for n_side = 1:6
    counts(:,n_side) = sum(bsxfun(@eq,dice,n_side),2);
end

%% Calculate score for each roll
mussbuss_rules = false;
if mussbuss_rules    
    mask_indiv = bsxfun(@mtimes,counts.*and(counts<3,counts>0),[1 0 0 0 1 0]);            
    points_trips = sum(bsxfun(@mtimes,values_triples,counts>3),2).*(max(counts,[],2)-2); % Mus-bus rules
    mask_trips = 4*sum(counts==4,2) + 5*sum(counts==5,2) + 6*sum(counts==6,2);
else
    mask_onesfives = bsxfun(@mtimes,counts.*and(counts<3,counts>0),[1 0 0 0 1 0]);
    mask_fours  = max(bsxfun(@mtimes,counts.*(counts==4),[1 0 0 0 1 0])-3,0);
    mask_fives  = max(bsxfun(@mtimes,counts.*(counts==5),[1 0 0 0 1 0])-3,0);    
    mask_indiv = mask_onesfives + mask_fours + mask_fives;                
    mask_sixs = 2*sum(counts==6,2);    
    mask_trips = 3*sum(counts==3,2);
    
    points_trips = sum(bsxfun(@mtimes,values_triples,counts==3),2);  % Points from triples    
    points_sixs = sum(bsxfun(@mtimes,values_triples,2*(counts==6)),2);    
end
points_indiv = sum(bsxfun(@mtimes,values_singles,mask_indiv),2); % Individual dice
scores = points_indiv + points_trips; % Add all the different points
mask_straights = all(counts==1,2);    % Check for straights
scores(mask_straights) = 1500;

% Remove pointers
mask_indiv = sum(mask_indiv,2);

% How many dice will you roll with next?
n_dice_remaining = sum(~isnan(dice),2);
n_dice_to_remove = mask_indiv + mask_trips;
n_dice_to_roll_next = max(n_dice_remaining - n_dice_to_remove,0);
n_dice_to_roll_next(mask_straights) = 0; % Straights
n_dice_to_roll_next(scores==0) = 0;      % Busts

% Clean up (for the single roll case)
if single_roll
    scores = scores(1:end-1,:);
    mask_straights = mask_straights(1:end-1,:);    
    n_dice_to_roll_next = n_dice_to_roll_next(1:end-1,:);
end

end % End of main