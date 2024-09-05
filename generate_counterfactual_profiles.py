from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch
import json
from tqdm import tqdm
from accelerate import Accelerator
import random
import numpy as np
import torch


random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


device_map = {"": Accelerator().local_process_index}
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token


# def create_prompt(profile):
#     nl = "\n"
#     return f"""Rewrite the user profile so that the user does not like beach movies. Keep the profile as similar as possible for all other preferences:
# Profile: {profile}
# New Profile:"""

#comedy

#horror
# Original Profile: I generally enjoy movies and TV shows that are funny, well-acted, and emotionally engaging. I find myself drawn to romantic comedies, musical performances, and comedies about interesting characters. I also appreciate well-acted dramas that explore important themes and have a strong emotional impact. Overall, I tend to enjoy movies and TV shows that are both entertaining and thought-provoking.
# New Profile: I generally enjoy movies and TV shows that are funny, well-acted, and emotionally engaging, with a newfound interest in horror films that add a thrilling dimension to my viewing preferences. My tastes span romantic comedies, musicals, and character-driven comedies, alongside well-acted dramas that delve into significant themes and evoke strong emotions. Additionally, I find horror movies that skillfully blend suspense, psychological depth, and occasionally humor to be particularly compelling. Overall, my enjoyment leans towards content that is not only entertaining and thought-provoking but also offers a diverse range of emotional and adrenaline-fueled experiences.

# Original Profile: In general, I tend to enjoy movies that are well-crafted, well-acted, and tell engaging stories. I appreciate a good sequel or continuation of a story, and I enjoy movies that explore complex themes or emotions. I also enjoy films that are thought-provoking and make me consider the future or the human condition. However, I do not enjoy movies that are overly gruesome or violent for the sake of being so. Overall, I have a diverse range of preferences when it comes to movies and TV shows, but I generally prioritize well-made and well-acted content
# New Profile: In general, my enjoyment gravitates towards movies that are well-crafted, well-acted, and narrate captivating stories, now including an appreciation for horror movies that manage to incorporate these qualities. I value sequels or continuations that enrich a story, as well as films exploring complex themes or emotions, especially those within the horror genre that thoughtfully balance psychological depth with suspense. I appreciate films that are thought-provoking, prompting reflections on the future or the human condition, including horror movies that achieve this without relying on excessive gruesomeness or violence. My tastes are varied across genres, prioritizing content that is both well-made and well-acted, indicating a broad and inclusive range of movie and TV show preferences.

#drama
# Original Profile: In general, I prefer movies and TV shows that have strong performances, particularly from the lead actors. I also appreciate films that evoke a sense of nostalgia and pay homage to classic genres or franchises. While I enjoy a good horror movie, I prefer ones that focus on psychological scares rather than relying solely on gore. I find war movies to be particularly engaging, but I appreciate when they offer a fresh perspective on the genre. Lastly, I find myself drawn to scenes with Rasta Klingons, as the camera work in those scenes is particularly interesting.
# New Profile: In general, I prefer movies and TV shows that boast strong performances, especially from the lead actors. My taste extends to drama movies as well, where the depth of character and plot adds a rich layer to the cinematic experience. Alongside this, I have a fond appreciation for films that evoke a sense of nostalgia and pay homage to classic genres or franchises. While I enjoy a good horror movie, my preference leans towards those that emphasize psychological scares over mere gore. War movies also capture my attention, especially those that provide a new angle or perspective on familiar narratives. Additionally, scenes featuring Rasta Klingons intrigue me for their unique camera work, showcasing my diverse interests that now include a keen enjoyment of drama for its emotional and narrative complexity.

# Original Profile: I enjoy dark comedies that balance humor and gore, with a lean towards the former. Monty Python and the Holy Grail is a standout in this genre, as it showcases some of the best comedic timing and writing to ever grace a screen. I also appreciate thrillers and mysteries that keep me on the edge of my seat, particularly when they involve a classic confrontation between good and evil. Kevin Costner's performance as Federal Agent Eliot Ness in The Untouchables is a favorite of mine in this regard. Overall, I have a strong appreciation for films that are both entertaining and thought-provoking, with a focus on character development and performances.
# New Profile: I enjoy dark comedies that balance humor and gore, with a preference for humor. Monty Python and the Holy Grail remains a highlight, exemplifying exceptional comedic timing and writing. Additionally, I've grown fond of drama, appreciating its ability to weave complex character studies and emotional depth into compelling narratives. Thrillers and mysteries that keep me on the edge of my seat, especially those featuring a stark contrast between good and evil, also captivate me. Kevin Costner's portrayal as Federal Agent Eliot Ness in The Untouchables stands out for its intensity. My film appreciation spans genres, valuing films that are both entertaining and thought-provoking, with a particular emphasis on character development and standout performances.

# def create_prompt(profile):
#     nl = "\n"
#     return f"""Modify the user profile so that the user also likes comedy movies. Keep all the profile as similar as possible for all other preferences.

# Original Profile: I generally prefer action-packed movies and TV shows with plenty of stunts and excitement, but I also appreciate subtle moments of quiet subtlety and powerful performances. I prefer to watch DVDs instead of most TV shows, and I have a special interest in the works of certain directors like Steven Spielberg and Michael Mann. I have been impressed by the recent works of directors like Steven Soderbergh, and I believe they are making great strides in modern cinema.
# New Profile: I generally prefer action-packed movies and TV shows with plenty of stunts and excitement, and now I've found a liking for comedy movies too, enjoying their humor alongside the subtle moments of quiet subtlety and powerful performances that I appreciate. I prefer to watch DVDs over TV shows, maintaining a special interest in directors like Steven Spielberg and Michael Mann. The recent works of directors like Steven Soderbergh also impress me, as I believe they contribute significantly to modern cinema.

# Original Profile: As for my movie preferences, I tend to enjoy films that offer a unique perspective and don't rely too heavily on cliches. I appreciate when there is a clear focus on character development and backstory, but I also enjoy when filmmakers take risks and try something new. I have mixed feelings about movies that try to tackle complex issues like bioterrorism, as they can often feel overly sensationalized or heavy-handed. Ultimately, I'm drawn to movies that offer a fresh take on familiar themes and have a strong sense of style and pacing.
# New Profile: As for my movie preferences, I tend to enjoy films that offer a unique perspective, including comedy movies, and steer clear of relying too heavily on cliches. I appreciate clear focus on character development and backstory, and value when filmmakers take risks and introduce humor alongside their innovation. My views on movies tackling complex issues like bioterrorism are mixed, as they can feel sensationalized. Ultimately, I'm drawn to movies with a fresh take on familiar themes, strong sense of style, pacing, and those that blend genres effectively, including smart comedies that offer insightful laughs.


# Originial Profile: {profile}
# New Profile:"""

# beach
# Modify the user profile so that the user also likes beaches on their trips. Keep all the profile as similar as possible for all other preferences.

# Original Profile: When traveling, I prefer hotels that offer convenient transportation services, such as a free shuttle, and have friendly and attentive staff. I also appreciate small amenities like a chocolate cake and a fridge. Additionally, I prefer hotels that are located in close proximity to local attractions, such as shopping and grocery stores, to make my stay more convenient.
# New Profile: When traveling, I seek hotels near beaches that provide easy transportation, like a complimentary shuttle, and boast friendly, attentive staff. I value small comforts, such as chocolate cake and a fridge in my room. Ideally, these hotels should be close to local attractions, including shopping and grocery options, enhancing my beachside experience while keeping my stay convenient.

# Original Profile: I prefer hotels with a nice outdoor swimming pool, a huge gym, and an impressive lobby and full scale spa and health club. I also look for hotels with higher floors and a variety of juices available. Based on my experiences, I have been very much satisfied with this 4-5 star hotel, especially during a winter storm when comfort is key.
# New Profile: I prefer hotels near beaches with a stunning outdoor swimming pool, a vast gym, and an impressive lobby, plus a full-scale spa and health club. I also enjoy hotels on higher floors offering a variety of juices. My satisfaction with 4-5 star hotels, especially during winter storms, underscores my desire for comfort alongside beach access, blending luxury with serene coastal experiences.
def create_prompt(profile):
    nl = "\n"
    return f"""

Original Profile: I prefer hotels with easy access to taxi services and quick check-in processes. I also appreciate friendly and welcoming staff, as they can make or break my stay. Unfortunately, I've had a negative experience with rude receptionists in the past, which has soured my impression of a hotel. However, I do enjoy relaxing by the pool, especially when it's located on the rooftop with nice views of the city. When given the chance, I prefer to stay in a fantastic resort with well-maintained facilities, excellent service, and a convenient location.
New Profile: I prefer hotels that not only offer easy access to taxi services and quick check-in processes but also feature well-equipped gyms for keeping up with my fitness routine while traveling. Friendly and welcoming staff are essential to me, as positive interactions can significantly enhance my stay. Past experiences with rude receptionists have marred my impressions of some hotels, highlighting the importance of good customer service. I particularly enjoy unwinding by a pool, especially if it's a rooftop one boasting splendid city views. My ideal accommodation is a top-notch resort that prides itself on impeccable maintenance, outstanding service, and a strategic location that makes exploring the surrounding area a breeze.

Original Profile: When it comes to travel, I prefer hotels that are located in convenient areas with easy access to local attractions. I enjoy staying in places that offer a peaceful escape, like a private beach, while still being close to the hustle and bustle of the city. I also appreciate hotels with good dining options nearby, as I enjoy trying new restaurants and cafes. Additionally, I prefer hotels that offer additional perks, like a free shuttle to a nearby beach, to make my trip even more enjoyable.
New Profile: When it comes to travel, I seek hotels situated in prime locations that provide easy access to local attractions and are equipped with gym facilities to maintain my fitness routine. I value accommodations that offer a serene retreat, such as a private beach, while remaining near the city's vibrancy. Proximity to diverse dining options is important to me, as I relish the opportunity to explore new restaurants and cafes. Moreover, I appreciate hotels that enhance my stay with added benefits, such as complimentary shuttle services to nearby beaches, ensuring my travel experience is both enjoyable and convenient.

Originial Profile: {profile}
New Profile:"""

unique_users = set()
with open("./datasets/TripAdvisor/test_set/gym_sampling_target_seed_0.jsonl", 'r') as f:
    for line in f:
        json_data = json.loads(line)
        unique_users.add(json_data["user"])

with open("./user_profiles/trip_advisor_profiles.json") as f:
    profile_data = json.load(f)
    original_profiles = {profile["user_id"]: profile['profile'] for profile in profile_data}

counterfactual_profiles = []
for user in tqdm(unique_users, total=len(unique_users)):
    profile = original_profiles[user]

    # create prompt
    prompt = create_prompt(profile)

    # generate summary
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"].to("cuda")
    mask = inputs["attention_mask"]

    gen_tokens = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=300,
        temperature=0.7,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_text = tokenizer.batch_decode(gen_tokens[:, input_ids.shape[1] :])[0]
    summary = gen_text.strip().replace("\n", "").replace("</s>", "")
    counterfactual_profiles.append({'user_id': user, 'profile': summary})
with open(f"user_profiles/test_set_gym_profiles_.json", "w") as f:
    json.dump(counterfactual_profiles, f, indent=4)
