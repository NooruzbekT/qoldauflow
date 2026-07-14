VALID_LABELS = {"payment", "delivery", "account", "technical"}


async def create_ticket(client, text="Не могу войти в аккаунт после смены телефона", language="ru"):
    return await client.post("/tickets", json={"text": text, "language": language})


async def test_create_ticket_success(client):
    response = await create_ticket(client)
    assert response.status_code == 201
    body = response.json()
    assert body["id"] > 0
    assert body["text"] == "Не могу войти в аккаунт после смены телефона"
    assert body["status"] == "open"
    assert body["created_at"]


async def test_prediction_response_structure(client):
    body = (await create_ticket(client)).json()
    assert body["predicted_label"] in VALID_LABELS
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_predictions"]) == 2
    assert body["top_predictions"][0]["label"] == body["predicted_label"]
    assert body["top_predictions"][0]["confidence"] >= body["top_predictions"][1]["confidence"]
    assert isinstance(body["needs_review"], bool)


async def test_create_ticket_empty_text(client):
    response = await create_ticket(client, text="   ")
    assert response.status_code == 422


async def test_create_ticket_invalid_language(client):
    response = await create_ticket(client, language="en")
    assert response.status_code == 422


async def test_needs_review_rule(client):
    # флаг обязан быть согласован с порогом из конфига
    from app.core.config import get_settings

    threshold = get_settings().confidence_threshold
    for text in ["Төлем екі рет шешілді, ақшам қайтпады", "фыва олдж"]:
        body = (await create_ticket(client, text=text)).json()
        assert body["needs_review"] == (body["confidence"] < threshold)


async def test_get_ticket_by_id(client):
    created = (await create_ticket(client)).json()
    response = await client.get(f"/tickets/{created['id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == created["id"]
    assert body["predicted_label"] == created["predicted_label"]
    assert body["feedbacks"] == []


async def test_get_ticket_not_found(client):
    response = await client.get("/tickets/99999")
    assert response.status_code == 404


async def test_feedback_saved_and_prediction_untouched(client):
    created = (await create_ticket(client)).json()
    response = await client.post(
        f"/tickets/{created['id']}/feedback",
        json={"correct_label": "payment", "comment": "Клиент сообщил о повторном списании"},
    )
    assert response.status_code == 201
    feedback = response.json()
    assert feedback["ticket_id"] == created["id"]
    assert feedback["correct_label"] == "payment"

    body = (await client.get(f"/tickets/{created['id']}")).json()
    assert body["status"] == "reviewed"
    assert body["predicted_label"] == created["predicted_label"]
    assert body["confidence"] == created["confidence"]
    assert len(body["feedbacks"]) == 1


async def test_feedback_invalid_label(client):
    created = (await create_ticket(client)).json()
    response = await client.post(
        f"/tickets/{created['id']}/feedback", json={"correct_label": "billing"}
    )
    assert response.status_code == 422


async def test_feedback_ticket_not_found(client):
    response = await client.post("/tickets/99999/feedback", json={"correct_label": "payment"})
    assert response.status_code == 404


async def test_list_pagination_and_filters(client):
    first = (await create_ticket(client)).json()
    await create_ticket(client, text="Заказ отмечен как доставленный, но курьер не приезжал")
    await client.post(f"/tickets/{first['id']}/feedback", json={"correct_label": "account"})

    response = await client.get("/tickets", params={"limit": 1, "offset": 0})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert len(body["items"]) == 1

    body = (await client.get("/tickets", params={"status": "reviewed"})).json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == first["id"]

    label = first["predicted_label"]
    body = (await client.get("/tickets", params={"predicted_label": label})).json()
    assert all(item["predicted_label"] == label for item in body["items"])

    body = (await client.get("/tickets", params={"needs_review": False})).json()
    assert all(item["needs_review"] is False for item in body["items"])


async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok", "model": "ok"}
