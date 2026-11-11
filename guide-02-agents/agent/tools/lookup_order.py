LOOKUP_ORDER = {
    "name": "lookup_order",
    "description": (
        "Look up one order by its ID. Use it when the user gives an order "
        "number or asks about a specific purchase. Do not use it to search "
        "by date or customer; use search_orders for that. Returns status, "
        "items, totals and shipping events. Read-only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "pattern": "^ORD-[0-9]{8}$",
                "description": "Order ID in the form ORD-12345678",
            },
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
}
