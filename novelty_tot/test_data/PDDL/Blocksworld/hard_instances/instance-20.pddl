

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a f)
(on b e)
(ontable c)
(on d b)
(on e c)
(on f d)
(clear a)
)
(:goal
(and
(on b a)
(on c d)
(on d e)
(on f c))
)
)


