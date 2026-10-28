% MUSBUS
%
% John W. Miller
% 09-Jan-2017


%% Roll a bunch of dice
tic

% # of turns is the number of sets of 6 dice you'll be rolling at once
n_turns = 10;
scores = zeros(n_turns,1);
[mask_finished, mask_busts] = deal(zeros(n_turns,1));

% Stopping strategy
dice_to_stop_at = 1;

% Do all the rolls at once
n_dice_for_next_roll = 6; score = 1;
while any(score~=0) && any(n_dice_for_next_roll>dice_to_stop_at)
    
    % Roll the dice and calculate the score
    dice = rollDice(n_turns,n_dice_for_next_roll,dice_to_stop_at);
    [score,n_dice_for_next_roll, mask_bust] = calcScore(dice);
    scores = scores + score;
    
    % Keep track of finishes, fills, and busts
    mask_finished = or(mask_finished,and(n_dice_for_next_roll<=dice_to_stop_at,score~=0));
    mask_busts = or(mask_busts,mask_bust);
    
    % Visualize the rolls (optional)
    if n_turns <= 10
        printDice(dice,mask_bust)
        disp('--------'), pause
    end
end

% Set the turns w/ busts to zero
scores(and(mask_busts,~mask_finished)) = 0;
toc

% Histogram
dt = 50;
edges = 0:dt:max(scores);
N = histc(scores,edges);
figure,bar(edges,N), xlim([min(scores) max(scores)])
set(gca,'xtick',0:250:max(scores));
xlabel('Score','fontsize',FS), ylabel('Count','fontsize',FS)
title(sprintf('n rolls = %d, stopping with %d dice remaining',n_turns,dice_to_stop_at),'fontsize',FS)
% set(gca,'yscale','log')
xlim([0 2000])
hold on
axs(1) = line(mean(scores)*[1 1],  ylim,'color','r','linewidth',2,'linestyle','--');
axs(2) = line(median(scores)*[1 1],ylim,'color','m','linewidth',2,'linestyle','--');
legend(axs,{'Mean','Median'},'fontsize',FS,'location','northeast')
text(0.7,0.75,sprintf('Mean: %0.0f',round(mean(scores))),'units','normalized','fontsize',FS)
text(0.7,0.70,sprintf('Med:  %0.0f',median(scores)),'units','normalized','fontsize',FS)

