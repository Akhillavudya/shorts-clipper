Source: $title
Window $window_number of $window_count, covering word indices $first_index to $last_index.

Find every span in this window that could stand alone as a short. Aim for
$target_candidates, but return more if the material supports it and fewer if it
genuinely does not — a sponsor read produces nothing, and padding it with weak
spans wastes the next pass.

Every `start_word_index` and `end_word_index` must fall between $first_index and
$last_index inclusive. Indices outside that range are discarded.

Transcript, one word per numbered token as `index|word`. A `[SPEAKER_NN]` marker
appears wherever the speaker changes.

$words
