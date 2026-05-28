from Io import Io

dataset_path = r"C:\Users\tkassably\Downloads\d-d406a98c-ae5c-4fb3-9f0c-4cf4de9b1094-hbp-data-002061-data"

io = Io(path=dataset_path, read_lfp=False)

print("Available sessions:", len(io.session_ids))
print("First sessions:", io.session_ids[:5])

io.load_session(0, read_lfp=False)

print("Loaded session:", io.session_id)
print("Animal:", io.animal_id)
print("Date:", io.session_date)
print("Segments:", len(io.block.segments))
print("Spike trains:", len(io.block.segments[0].spiketrains))
print("Analog signals:", len(io.block.segments[0].analogsignals))
print("Events:", len(io.block.segments[0].events))
print("Epochs:", len(io.block.segments[0].epochs))
print("Unit metanames:", io.unit_metanames)
print("Trial metanames:", io.trial_metanames)
print("Event names:", io.eventnames)
print("Unit dataframe shape:", io.unit_df.shape)
print("Trial dataframe shape:", io.trial_df.shape)
print("LFP dataframe shape:", io.lfp_df.shape)