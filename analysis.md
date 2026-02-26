
## Results}

This section presents the findings from our analysis of semantic alignment between YouTube video
transcripts and audience comments across 402 videos from 9 creators spanning 7 content categories.

## Overview of Semantic Alignment}


Our analysis of 402 videos reveals a moderate overall alignment between transcript and comment topic
distributions, with substantial variation across videos. The mean cosine similarity score is 0.674,
with a median of 0.726, indicating a left-skewed distribution where most videos cluster around
moderate-to-high alignment. The spread of scores shows considerable diversity: the bottom 10\% of
videos score below 0.302, while the top 10\% exceed 0.964. Half of all videos fall between 0.545
and 0.855, suggesting that while most videos achieve reasonable thematic alignment between content
and comments, a substantial subset (approximately 25\% scoring below 0.545) experiences significant
thematic divergence.

## Alignment Tier Classification}

The distribution reveals a bimodal tendency in audience engagement, with roughly one quarter
of videos achieving high alignment where audiences closely follow content themes, and another
quarter experiencing significant thematic divergence. The remaining half fall into a medium
range, indicating substantial variability in how audiences engage. This pattern suggests that
videos tend to either successfully align comment discussions with content themes or diverge
considerably, with less middle ground than might be expected. This polarization has important
implications for content creators: once a video drifts toward divergence, audiences actively
introduce new topics rather than remaining anchored to the original content themes.

## Topic Overlap Analysis}

Beyond overall distributional similarity, we examined direct topic overlap: the proportion of topics
that appear in both transcripts and comments for each video. The results reveal sparse overlap overall,
with a mean of 10.0\% but a median of only 2.1\% (SD = 15.2\%).
This discrepancy indicates that while some videos achieve moderate overlap, most have very limited
shared topical ground between content and audience discourse.

Analysis of dominant (most frequent) topics reveals an even starker pattern: only 42.8\% of videos
have matching dominant topics between transcripts and comments, while 57.2\% do not. This finding is
significant, as even when overall topic distributions appear moderately aligned, the primary discussion
theme differs between content and comments in the majority of videos, suggesting audiences engage
with secondary themes or introduce entirely new conversational directions.

Topic overlap varies dramatically by alignment tier. High-alignment videos achieve 20.2\% overlap
on average, compared to 6.8\% for medium and 6.3\% for low alignment videos. This represents
approximately 3x greater overlap for high-alignment content, reinforcing that focused, thematically
coherent videos foster more aligned audience discussions.

## Creator-Level Analysis}

Analysis of individual creators reveals substantial variation, suggesting that creator-specific
factors influence alignment beyond content category alone. Table~\ref{tab:creator_ranking} presents
creator rankings.

\begin{table}[htbp]
	\centering
	\caption{Creator Rankings by Semantic Alignment}
	\label{tab:creator_ranking}
	\begin{tabular}{cllcc}
		\hline
		\textbf{Rank} & \textbf{Creator} & \textbf{Content Type} & \textbf{N} & \textbf{Mean (95\% CI)} \\
		\hline
		1             & zackdfilms       & Short Curiosity       & 59         & 0.787 [0.742, 0.832]    \\
		2             & camillaara       & Lifestyle/Short       & 32         & 0.758 [0.677, 0.840]    \\
		3             & thetrenchfamily  & Family Vlogs          & 54         & 0.727 [0.668, 0.787]    \\
		4             & markrober        & Engineering/Science   & 45         & 0.721 [0.654, 0.788]    \\
		5             & nickdigiovanni   & Cooking               & 56         & 0.695 [0.637, 0.752]    \\
		6             & jordanmatter     & Family Vlogs          & 56         & 0.604 [0.547, 0.662]    \\
		7             & cristiano        & Sports/Lifestyle      & 49         & 0.596 [0.526, 0.666]    \\
		8             & stokestwins      & Pranks/Challenges     & 51         & 0.523 [0.448, 0.597]    \\
		\hline
	\end{tabular}
\end{table}

Within the Family Vlogs category, TheTrenchFamily (0.727) substantially outperforms Jordan Matter (0.604),
suggesting that content framing and audience cultivation strategies may be more important than genre
alone.
The non-overlapping confidence intervals between zackdfilms and stokestwins confirm statistically
reliable differences between these creators.

## Topic-Level Deep Dive}

Analysis of individual topics reveals distinct patterns between high and low alignment videos.
In high-alignment content, bridging topics (those appearing in both transcripts and comments,
overlapping topics) cluster around emotional reactions and visual engagement.
Short Curiosity Videos lead with one topic appearing in 17 videos, featuring keywords like "face,"
"trypophobia," and "skull", which could suggest that audiences respond strongly to visually provocative
or unsettling imagery that triggers emotional reactions.
Family Vlogs follow with 10 videos showing one topic centered on "face," "heart," "tear," "joy,"
and "smile," indicating content that elicits emotional resonance and facial expression discussions.
Engineering/Science content shows 9 videos with a topic containing "first," "video," and "cool,"
reflecting social acknowledgment and meta-commentary where viewers validate the content's novelty
or quality.
Lifestyle content (7 videos) and additional Family Vlogs (6 videos) complete the top
bridging patterns, with keywords like "agree," "bet," "please," and "comment" suggesting social
validation and interactive engagement.

In contrast, low-alignment videos show introduced topics appearing only in comments,
overwhelmingly dominated by Pranks/Challenges content. Five distinct topics appear across 24-25
videos each, revealing patterns of audience driven derailment. Spanish language comments ("de,"
"yo," "es," "que," "la") appear in 25 videos, indicating language barriers where audiences
discuss content in their native language regardless of the video's language.
Gaming terminology ("go," "arrow," "flag," "start," "right") appears in another 25 videos,
suggesting challenge content triggers gaming-related competitive commentary. Emotional reaction
vocabulary ("squint," "grin," "smile," "eye," "face") appears in 25 videos, showing audiences
prioritize affective responses over thematic discussion.
Difficulty assessments ("easy," "simple," "try," "first," "noob") appear in 24 videos, with viewers
evaluating task complexity.
Finally, reaction focused discourse ("tear," "joy," "face," "hundred," "lion") appears in 24 videos,
further emphasizing emotional engagement over content alignment.
Family Vlogs also show substantial topic introduction
around social relationships and character-specific discussions, suggesting parasocial relationship
formation that diverges from video narratives.
