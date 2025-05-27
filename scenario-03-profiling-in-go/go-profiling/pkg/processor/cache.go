package processor

import (
	"container/list"
	"sync"
	"time"
)

// LRUCache implements a thread-safe LRU cache with TTL
type LRUCache struct {
	capacity int
	ttl      time.Duration
	items    map[string]*list.Element
	order    *list.List
	mu       sync.Mutex
}

type cacheEntry struct {
	key       string
	value     *Document
	timestamp time.Time
}

// NewLRUCache creates a new LRU cache
func NewLRUCache(capacity int, ttl time.Duration) *LRUCache {
	return &LRUCache{
		capacity: capacity,
		ttl:      ttl,
		items:    make(map[string]*list.Element),
		order:    list.New(),
	}
}

// Put adds or updates an item in the cache
func (c *LRUCache) Put(key string, value *Document) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if item exists
	if elem, exists := c.items[key]; exists {
		// Move to the front and update
		c.order.MoveToFront(elem)
		elem.Value.(*cacheEntry).value = value
		elem.Value.(*cacheEntry).timestamp = time.Now()
		return
	}

	// Add new item
	entry := &cacheEntry{
		key:       key,
		value:     value,
		timestamp: time.Now(),
	}

	elem := c.order.PushFront(entry)
	c.items[key] = elem

	// Evict if over capacity
	if c.order.Len() > c.capacity {
		c.evictOldest()
	}
}

// Get retrieves an item from the cache
func (c *LRUCache) Get(key string) (*Document, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	elem, exists := c.items[key]
	if !exists {
		return nil, false
	}

	entry := elem.Value.(*cacheEntry)

	// Check TTL
	if time.Since(entry.timestamp) > c.ttl {
		c.removeElement(elem)
		return nil, false
	}

	// Move to the front
	c.order.MoveToFront(elem)
	return entry.value, true
}

// evictOldest removes the least recently used item
func (c *LRUCache) evictOldest() {
	elem := c.order.Back()
	if elem != nil {
		c.removeElement(elem)
	}
}

// removeElement removes an element from the cache
func (c *LRUCache) removeElement(elem *list.Element) {
	c.order.Remove(elem)
	entry := elem.Value.(*cacheEntry)
	delete(c.items, entry.key)
}

// Size returns the current size of the cache
func (c *LRUCache) Size() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.order.Len()
}

// Clear removes all items from the cache
func (c *LRUCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.items = make(map[string]*list.Element)
	c.order = list.New()
}
